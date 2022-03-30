import networkx as nx
import pandas as pd
import itertools as it


# Creation of a Graph from natural language annotated debates
def debate_to_graph(debate):
    DG = nx.DiGraph()
    colorsfav = ['lightsteelblue', 'lightskyblue', 'steelblue', 'mediumblue', 'midnightblue']
    colorsaga = ['thistle', 'violet', 'mediumorchid', 'darkviolet', 'purple']
    for index, row in debate.iterrows():
        if str(row['TYPE (Part + Person)']) == 'INTRO':
            size = 1

        elif str(row['TYPE (Part + Person)']) == 'CONC':
            size = 100

        else:
            size = 5

        if row['TEAM STANCE'] == 'AGAINST':
            if pd.isna(row['ARGUMENT NUMBER']):
                DG.add_node(row['ID (Chronological)'], team=row['TEAM STANCE'], text_cat=row['ADU_CAT'], text_es=row['ADU_ES'], text_en=row['ADU_EN'], color=colorsaga[0], node_size=size)
            else:
                try:
                    DG.add_node(row['ID (Chronological)'], team=row['TEAM STANCE'], text_cat=row['ADU_CAT'], text_es=row['ADU_ES'], text_en=row['ADU_EN'], color=colorsaga[int(row['ARGUMENT NUMBER'])], node_size=size)
                except:
                    DG.add_node(row['ID (Chronological)'], team=row['TEAM STANCE'], text_cat=row['ADU_CAT'], text_es=row['ADU_ES'], text_en=row['ADU_EN'], color=colorsaga[0], node_size=size)

        elif row['TEAM STANCE'] == 'FAVOUR':
            if pd.isna(row['ARGUMENT NUMBER']):
                DG.add_node(row['ID (Chronological)'], team=row['TEAM STANCE'], text_cat=row['ADU_CAT'], text_es=row['ADU_ES'], text_en=row['ADU_EN'], color=colorsfav[0], node_size=size)
            else:
                try:
                    DG.add_node(row['ID (Chronological)'], team=row['TEAM STANCE'], text_cat=row['ADU_CAT'], text_es=row['ADU_ES'], text_en=row['ADU_EN'], color=colorsfav[int(row['ARGUMENT NUMBER'])], node_size=size)
                except:
                    DG.add_node(row['ID (Chronological)'], team=row['TEAM STANCE'], text_cat=row['ADU_CAT'], text_es=row['ADU_ES'], text_en=row['ADU_EN'], color=colorsfav[0], node_size=size)

        if not pd.isna(row['RELATED ID']):
            try:
                id_list = str(int(row['RELATED ID'])).split(';')

            except:
                id_list = str(row['RELATED ID']).split(';')

            for ident in id_list:
                if row['ARGUMENTAL RELATION  TYPE'] == 'RA':
                    DG.add_edge(row['ID (Chronological)'], int(ident), type=row['ARGUMENTAL RELATION  TYPE'], color='g')
                elif row['ARGUMENTAL RELATION  TYPE'] == 'CA':
                    DG.add_edge(row['ID (Chronological)'], int(ident), type=row['ARGUMENTAL RELATION  TYPE'], color='r')
                else:
                    DG.add_edge(row['ID (Chronological)'], int(ident), type=row['ARGUMENTAL RELATION  TYPE'], color='y')
        try:
            if not pd.isna(row['RELATED ID.1']):
                try:
                    id_list = str(int(row['RELATED ID.1'])).split(';')

                except:
                    id_list = str(row['RELATED ID.1']).split(';')

                for ident in id_list:
                    if row['ARGUMENTAL RELATION  TYPE.1'] == 'RA':
                        DG.add_edge(row['ID (Chronological)'], int(ident), type=row['ARGUMENTAL RELATION  TYPE.1'],
                                    color='g')
                    elif row['ARGUMENTAL RELATION  TYPE.1'] == 'CA':
                        DG.add_edge(row['ID (Chronological)'], int(ident), type=row['ARGUMENTAL RELATION  TYPE.1'],
                                    color='r')
                    else:
                        DG.add_edge(row['ID (Chronological)'], int(ident), type=row['ARGUMENTAL RELATION  TYPE.1'],
                                    color='y')
        except:
            pass

        try:
            if not pd.isna(row['RELATED ID.2']):
                try:
                    id_list = str(int(row['RELATED ID.2'])).split(';')
                except:
                    id_list = str(row['RELATED ID.2']).split(';')

                for ident in id_list:
                    if row['ARGUMENTAL RELATION  TYPE.2'] == 'RA':
                        DG.add_edge(row['ID (Chronological)'], int(ident), type=row['ARGUMENTAL RELATION  TYPE.2'],
                                    color='g')
                    elif row['ARGUMENTAL RELATION  TYPE.2'] == 'CA':
                        DG.add_edge(row['ID (Chronological)'], int(ident), type=row['ARGUMENTAL RELATION  TYPE.2'],
                                    color='r')
                    else:
                        DG.add_edge(row['ID (Chronological)'], int(ident), type=row['ARGUMENTAL RELATION  TYPE.2'],
                                    color='y')
        except:
            pass

    return DG


# Generate a Dung's Abstract Argumentation Framework from natural language argumentation graphs
def graph_to_af(graph):

    # Remove same-stance adu negations
    remove_attacks = [(u, v) for u, v in graph.edges() if graph.nodes[u]['team'] == graph.nodes[v]['team'] and graph[u][v]['type'] == 'CA']

    # Remove different stance adu inferences and rephrases
    remove_supports = [(u, v) for u, v in graph.edges() if graph.nodes[u]['team'] != graph.nodes[v]['team'] and (graph[u][v]['type'] == 'RA' or graph[u][v]['type'] == 'MA')]

    graph.remove_edges_from(remove_attacks)
    graph.remove_edges_from(remove_supports)

    # Remaining attacks to be rebuilt in the AF
    attacks = [(u, v) for u, v in graph.edges() if graph[u][v]['type'] == 'CA']
    graph.remove_edges_from(attacks)

    und_g = graph.to_undirected()
    AF = nx.DiGraph()
    components = nx.connected_components(und_g)
    id = 0
    for sub_g in components:
        text_cat = ''
        text_es = ''
        text_en = ''
        node_set = []
        team = None
        for node in sub_g:
            if und_g.nodes[node]['node_size'] == 100 or und_g.nodes[node]['node_size'] == 1:
            # if und_g.nodes[node]['node_size'] == 100:
                pass
            else:
                team = und_g.nodes[node]['team']
                text_cat += und_g.nodes[node]['text_cat']+' '
                text_es += und_g.nodes[node]['text_es']+' '
                text_en += und_g.nodes[node]['text_en']+' '
                node_set.append(node)
            if team == 'FAVOUR':
                color = 'g'
            else:
                color = 'r'

        if len(node_set) > 0:
            AF.add_node(id, team=team, text_cat=text_cat, text_es=text_es, text_en=text_en, node_set=node_set, color=color, node_size=len(node_set))
            id += 1

    unremovable_nodes = []
    for node in AF.nodes:
        for attack in attacks:
            if attack[0] in AF.nodes[node]['node_set']:
                for node2 in AF.nodes:
                    if attack[1] in AF.nodes[node2]['node_set']:
                        AF.add_edge(node, node2)
                        unremovable_nodes.append(node)
                        unremovable_nodes.append(node2)
    rmv = []
    for node in AF.nodes:
        if node not in unremovable_nodes and len(AF.nodes[node]['node_set']) == 1:
        # if node not in unremovable_nodes:
            rmv.append(node)

    AF.remove_nodes_from(rmv)

    return AF


def naive_semantics(framework):
    acceptable = []
    isolated_nodes = list(nx.isolates(framework))
    naf = framework.copy()
    naf.remove_nodes_from(isolated_nodes)
    node_list = list(naf.nodes)
    n = len(node_list)
    found = False
    while n > 0:
        extensions = it.combinations(node_list, n)
        for ext in extensions:
            sg = naf.subgraph(list(ext))
            # check conflict-free property
            if len(list(sg.edges)) == 0:
                acceptable.append(isolated_nodes+list(ext))
                found = True
        if found:
            break
        else:
            n -= 1

    if len(isolated_nodes) > 0 and len(acceptable) == 0:
        acceptable = [isolated_nodes]
    return acceptable


def preferred_semantics(framework, n):
    acceptable = []
    isolated_nodes = list(nx.isolates(framework))
    paf = framework.copy()
    paf.remove_nodes_from(isolated_nodes)
    node_list = list(paf.nodes)
    found = False
    while n > 0:
        extensions = it.combinations(node_list, n)
        for ext in extensions:
            sg = paf.subgraph(list(ext))
            # check conflict-free property
            if len(list(sg.edges)) == 0:
                adm = True
                # check admissibility
                for argument in ext:
                    attacked = False
                    for attack in framework.edges:
                        if attack[1] == argument:
                            attacked = True
                            defended = False
                            attacker = attack[0]
                            for att in framework.edges:
                                if attacker == att[1] and att[0] in ext:
                                    defended = True

                    if attacked and not defended:
                        adm = False
                        break
                if adm:
                    acceptable.append(isolated_nodes + list(ext))
                    found = True
        if found:
            break
        else:
            n -= 1

    if len(isolated_nodes) > 0 and len(acceptable) == 0:
        acceptable = [isolated_nodes]
    return acceptable


def extension_to_sample_graph(extension, framework):
    favour_nodes = []
    against_nodes = []
    sample_nxgraph = nx.DiGraph(framework.subgraph(extension))
    for node in extension:
        if framework.nodes[node]['team'] == 'FAVOUR':
            favour_nodes.append(node)
        elif framework.nodes[node]['team'] == 'AGAINST':
            against_nodes.append(node)

    for fnode in favour_nodes:
        for anode in against_nodes:
            sample_nxgraph.add_edge(fnode, anode)
            sample_nxgraph.add_edge(anode, fnode)

    return sample_nxgraph
