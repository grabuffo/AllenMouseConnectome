from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import numpy as np
import copy

def mcc(resolution=100):
    """
    The function imports the mouse connectivity cache
    """
    return MouseConnectivityCache(resolution=resolution)


def dictionary_builder(mcc, structure_tree, ROIs, transgenic_line):
    """
    The function creates a dictionary with information about which experiments need to be downloaded
    """
    #define map 
    ia_map = structure_tree.get_id_acronym_map()
    # open up a list of all of the experiments
    all_experiments = mcc.get_experiments(dataframe=True, cre=transgenic_line)
    # build dict of injection structure id to experiment list
    ist2e = {}
    for eid in all_experiments.index:
        sab = all_experiments.loc[eid]['structure_abbrev']
        if sab in ROIs:
            isti = ia_map[sab]
            if isti not in ist2e:
                ist2e[isti] = []
            ist2e[isti].append(eid)
        else:
            ist = all_experiments.loc[eid]['injection_structures']
            for isti in ist:
                if isti not in ist2e:
                    ist2e[isti] = []
                ist2e[isti].append(eid)
    return ist2e


def download_an_construct_matrix(mcc, weighting, ist2e, transgenic_line):
    """
    The function downloads experiments necessary to build the connectivity and returns the projection maps
    """
    projmaps = {}
    if weighting == 3:  # download projection energy
        for isti, elist in ist2e.items():
            projmaps[isti] = mcc.get_projection_matrix(
                experiment_ids=elist,
                projection_structure_ids=list(ist2e),  # summary_structure_ids,
                parameter='projection_energy')
            print('injection site id', isti, ' has ', len(elist), ' experiments with pm shape ',
                        projmaps[isti]['matrix'].shape)
    else:  # download projection density:
        eli=[]
        for isti, elist in ist2e.items():
            projmaps[isti] = mcc.get_projection_matrix(
                experiment_ids=elist,
                projection_structure_ids=list(ist2e),  # summary_structure_ids,
                parameter='projection_density')
            eli=list(set(eli + elist))
            print('injection site id', isti, ' has ', len(elist), ' experiments with pm shape ',
                        projmaps[isti]['matrix'].shape)
        if weighting == 1:  # download injection density
            injdensity = {}
            #all_experiments = mcc.get_experiments(dataframe=True, cre=transgenic_line)
            for exp_id in eli:#all_experiments['id']:
                inj_d = mcc.get_injection_density(exp_id, file_name=None)
                # all the experiments have only an injection sites (only 3 coordinates),
                # thus it is possible to sum the injection matrix
                injdensity[exp_id] = (np.sum(inj_d[0]) / np.count_nonzero(inj_d[0]))
                print('Experiment id', exp_id, ', the total injection density is ', injdensity[exp_id])
            # in this case projmaps will contain PD/ID
            for inj_id in range(len(list(projmaps.values()))):
                index = 0
                for exp_id in list(projmaps.values())[inj_id]['rows']:
                    list(projmaps.values())[inj_id]['matrix'][index] = list(projmaps.values())[inj_id]['matrix'][index] / \
                                                                 injdensity[exp_id]
                    index += 1
    return projmaps


# the function 'pms_cleaner' 
def pms_cleaner(input_projmaps):
    """
    the method cleans the file projmaps in 2 steps: 
    1) Set all the target sites to be the same for all the injection sites
    2) Remove Nan
    """
    projmaps_clean1={}
    # 1) Set all the target sites to be the same for all the injection sites
    for inj_id in input_projmaps:
        c=input_projmaps[inj_id]['columns']
        m=input_projmaps[inj_id]['matrix']
        excl=[]
        for ic in range(len(c)):
            if c[ic]['structure_id'] not in input_projmaps.keys():
                excl.append(ic)
        for icc in excl[::-1]:
            c.pop(icc)
        m=np.delete(m,excl,axis=1)
        projmaps_clean1[inj_id]={'columns':c,'matrix':m,'rows':input_projmaps[inj_id]['rows']}
    print('In total %d injection sites have been removed after imposing that targets and injection sites are the same. New number of keys in projmaps is N=%d'%(len(input_projmaps.keys())-len(projmaps_clean1.keys()),len(projmaps_clean1.keys())))
    
    projmaps={}
    # 2) Remove Nan
    for inj_id in projmaps_clean1:
        c=projmaps_clean1[inj_id]['columns']
        m=projmaps_clean1[inj_id]['matrix']
        excl=[]
        for ic in range(len(c)):
            if 1 in np.isnan(m[:,ic]):
                excl.append(ic)
        for icc in excl[::-1]:
            c.pop(icc)
        m=np.delete(m,excl,axis=1)
        projmaps[inj_id]={'columns':c,'matrix':m,'rows':projmaps_clean1[inj_id]['rows']}
    print('In total %d injection sites have been removed after imposing that there is no Nan. New number of keys in projmaps is N=%d'%(len(projmaps_clean1.keys())-len(projmaps.keys()),len(projmaps.keys())))
    return projmaps

def areas_volume_threshold(mcc, projmaps_old, vol_thresh, resolution):
    """
    the method includes in the parcellation only brain regions whose volume is greater than vol_thresh
    """
    threshold = vol_thresh / (resolution ** 3)
    id_all=list(projmaps_old.keys())
    id_ok = []
    
    projmaps_rmvol1=copy.deepcopy(projmaps_old)
    for ID in projmaps_old:
        mask, _ = mcc.get_structure_mask(ID)
        tot_voxels = int((np.count_nonzero(mask)) / 2)  # mask contains both left and right hemisphere
        if tot_voxels > threshold:
            id_ok.append(ID)
    print('Regions removed because too small')
    print(list(set(id_all)-set(id_ok)))
    
    print('Big regions left')
    print(id_ok)
    
    # Remove keys of all areas that are not in id_ok from the injection list
    for checkid in id_all:
        if checkid not in id_ok:
            projmaps_rmvol1.pop(checkid, None)
            
    # Remove areas that are not in id_ok from target list (columns+matrix)
    projmaps_rmvol={}
    for inj_id in projmaps_rmvol1:
        c=projmaps_rmvol1[inj_id]['columns']
        m=projmaps_rmvol1[inj_id]['matrix']
        excl=[]
        for ic in range(len(c)):
            if c[ic]['structure_id'] not in id_ok:
                excl.append(ic)
        for icc in excl[::-1]:
            c.pop(icc)
        m=np.delete(m,excl,axis=1)
        projmaps_rmvol[inj_id]={'columns':c,'matrix':m,'rows':projmaps_rmvol1[inj_id]['rows']}
    return projmaps_rmvol

def create_file_order(projmaps, structure_tree):
    """
    the method creates file order and keyord that will be the link between the structural conn
    order and the id key in the Allen database
    """
    order = {}
    for target_id in projmaps.keys():
        order[structure_tree.get_structures_by_id([target_id])[0]['graph_order']] = [target_id]
        order[structure_tree.get_structures_by_id([target_id])[0]['graph_order']].append(
            structure_tree.get_structures_by_id([target_id])[0]['acronym'])
        order[structure_tree.get_structures_by_id([target_id])[0]['graph_order']].append(
            structure_tree.get_structures_by_id([target_id])[0]['name'])#name
    key_ord = list(order)
    key_ord.sort()
    return order, key_ord

def mouse_brain_visualizer(vol, order, key_ord, structure_tree, projmaps):
    """
    the method returns a volume indexed between 0 and N-1, with N=tot brain areas in the parcellation.
    -1=background and areas that are not in the parcellation
    """
    tot_areas = len(key_ord) * 2
    indexed_vec = np.arange(tot_areas).reshape(tot_areas, )
    # vec indexed between 0 and (N-1), with N=total number of area in the parcellation
    indexed_vec = indexed_vec + 1  # vec indexed between 1 and N
    indexed_vec = indexed_vec * (10 ** (-(1 + int(np.log10(tot_areas)))))
    # vec indexed between 0 and 0,N (now all the entries of vec_indexed are < 1 in order to not create confusion
    # with the entry of Vol (always greater than 1)
    vol_r = vol[:, :, :int(vol.shape[2] / 2)]
    vol_r = vol_r.astype(np.float64)
    vol_l = vol[:, :, int(vol.shape[2] / 2):]
    vol_l = vol_l.astype(np.float64)
    index_vec = 0  # this is the index of the vector
    left = int(len(indexed_vec) / 2)
    for graph_ord_inj in key_ord:
        node_id = order[graph_ord_inj][0]
        if node_id in vol_r:  # check if the area is in the annotation volume
            vol_r[vol_r == node_id] = indexed_vec[index_vec]
            vol_l[vol_l == node_id] = indexed_vec[index_vec + left]
        child = []
        for ii in range(len(structure_tree.children([node_id])[0])):
            child.append(structure_tree.children([node_id])[0][ii]['id'])
        while len(child) != 0:
            if (child[0] in vol_r) and (child[0] not in list(projmaps)):
                vol_r[vol_r == child[0]] = indexed_vec[index_vec]
                vol_l[vol_l == child[0]] = indexed_vec[index_vec + left]
            child.remove(child[0])
        index_vec += 1  # index of vector
    vol_parcel = np.concatenate((vol_r, vol_l), axis=2)
    
    vol_parcel[vol_parcel >= 1] = 0  # set all the areas not in the parcellation to 0 since the background is zero
    vol_parcel = vol_parcel * (10 ** (1 + int(np.log10(tot_areas))))  # return to indexed between
    # 1 and N (with N=tot number of areas in the parcellation)
    vol_parcel = vol_parcel - 1  # with this operation background and areas not in parcellation will be -1
    # and all the others with the indexed between 0 and N-1
    vol_parcel = np.round(vol_parcel)
    vol_parcel = rotate_reference(vol_parcel)
    return vol_parcel

def rotate_reference(allen):
    """
    'rotate_reference' rotate the Allen 3D (x1,y1,z1) reference 
     in the TVB (https://www.thevirtualbrain.org/tvb/zwei) 3D reference (x2,y2,z2).
    The relation between the different reference system is: x1=z2, y1=x2, z1=y2
    """
    # first rotation in order to obtain: x1=x2, y1=z2, z1=y2
    vol_trans = np.zeros((allen.shape[0], allen.shape[2], allen.shape[1]), dtype=float)
    for x in range(allen.shape[0]):
        vol_trans[x, :, :] = (allen[x, :, :][::-1]).transpose()

    # second rotation in order to obtain: x1=z2, y1=x1, z1=y2
    allen_rotate = np.zeros((allen.shape[2], allen.shape[0], allen.shape[1]), dtype=float)
    for y in range(allen.shape[1]):
        allen_rotate[:, :, y] = (vol_trans[:, :, y]).transpose()
    return allen_rotate


def construct_structural_conn(projmaps, order, key_ord):
    """
    the method builds the Structural Connectivity (SC) matrix
    """
    len_right = len(list(projmaps))
    structural_conn = np.zeros((len_right, 2 * len_right), dtype=float)
    row = -1
    for graph_ord_inj in key_ord:
        row += 1
        inj_id = order[graph_ord_inj][0]
        columns = projmaps[inj_id]['columns']
        matrix = projmaps[inj_id]['matrix']
        # average on the experiments (NB: if there are NaN values not average!)
        if np.isnan(np.sum(matrix)):
            print('There is a Nan in matrix of injection structure %d'%inj_id)
        else:
            matrix = np.mean(matrix,axis=0) #(np.array([sum(matrix[:, i]) for i in range(matrix.shape[1])]) / (matrix.shape[0]))
        # order the target
        col = -1
        for graph_ord_targ in key_ord:
            col += 1
            targ_id = order[graph_ord_targ][0]
            for index in range(len(columns)):
                if columns[index]['structure_id'] == targ_id:
                    if columns[index]['hemisphere_id'] == 2:
                        structural_conn[row, col] = matrix[index]
                    if columns[index]['hemisphere_id'] == 1:
                        structural_conn[row, col + len_right] = matrix[index]
    # save the complete matrix (both left and right inj):
    first_quarter = structural_conn[:, :int(structural_conn.shape[1] / 2)]
    second_quarter = structural_conn[:, int(structural_conn.shape[1] / 2):]
    sc_down = np.concatenate((second_quarter, first_quarter), axis=1)
    structural_conn = np.concatenate((structural_conn, sc_down), axis=0)
    structural_conn = structural_conn / (np.amax(structural_conn))  # normalize the matrix
    return structural_conn


def construct_centres(mcc, order, key_ord):
    """
    the method returns the centres of the brain areas in the selected parcellation
    """
    centres = np.zeros((len(key_ord) * 2, 3), dtype=float)
    names = []
    acro = []
    row = -1
    for graph_ord_inj in key_ord:
        node_id = order[graph_ord_inj][0]
        coord = [0, 0, 0]
        mask, _ = mcc.get_structure_mask(node_id)
        mask = rotate_reference(mask)
        mask_r = mask[:int(mask.shape[0] / 2), :, :]
        xyz = np.where(mask_r)
        if xyz[0].shape[0] > 0:  # Check if the area is in the annotation volume
            coord[0] = np.mean(xyz[0])
            coord[1] = np.mean(xyz[1])
            coord[2] = np.mean(xyz[2])
        row += 1
        centres[row, :] = coord
        coord[0] = (mask.shape[0]) - coord[0]
        centres[row + len(key_ord), :] = coord
        n = order[graph_ord_inj][2]
        N = order[graph_ord_inj][1]
        right = 'Right '
        names.append(str(right+n))
        acro.append(str(right+N))
    for graph_ord_inj in key_ord:
        n = order[graph_ord_inj][2]
        N = order[graph_ord_inj][1]        
        left = 'Left '
        names.append(str(left+n))
        acro.append(str(left+N))
    return centres, names, acro



def construct_tract_lengths(centres):
    """
    the method returns the tract lengths between the brain areas in the selected parcellation
    """
    len_right = int(len(centres) / 2)
    tracts = np.zeros((len_right, len(centres)), dtype=float)
    for inj in range(len_right):
        center_inj = centres[inj]
        for targ in range(len_right):
            targ_r = centres[targ]
            targ_l = centres[targ + len_right]
            tracts[inj, targ] = np.sqrt(
                (center_inj[0] - targ_r[0]) ** 2 + (center_inj[1] - targ_r[1]) ** 2 + (center_inj[2] - targ_r[2]) ** 2)
            tracts[inj, targ + len_right] = np.sqrt(
                (center_inj[0] - targ_l[0]) ** 2 + (center_inj[1] - targ_l[1]) ** 2 + (center_inj[2] - targ_l[2]) ** 2)
    # Save the complete matrix (both left and right inj):
    first_quarter = tracts[:, :int(tracts.shape[1] / 2)]
    second_quarter = tracts[:, int(tracts.shape[1] / 2):]
    tracts_down = np.concatenate((second_quarter, first_quarter), axis=1)
    tracts = np.concatenate((tracts, tracts_down), axis=0)
    return tracts