use super::*;

fn node(kind: DepKind, first: u64, second: u64) -> DepNode {
    DepNode { kind, key_fingerprint: Fingerprint::new(first, second).into() }
}

fn graph(nodes: Vec<DepNode>) -> SerializedDepGraph {
    let nodes = IndexVec::from_raw(nodes);
    let mut nodes_by_kind = Vec::new();
    let kinds = (0..DepKind::NUM_VARIANTS)
        .map(|kind| {
            let start = nodes_by_kind.len();
            nodes_by_kind.extend(nodes.iter_enumerated().filter_map(|(index, node)| {
                (node.kind != DepKind::Null && node.kind.as_usize() == kind).then_some(Some(index))
            }));
            LazyKindIndex {
                start: start as u32,
                len: (nodes_by_kind.len() - start) as u32,
                map: OnceLock::new(),
            }
        })
        .collect();
    SerializedDepGraph {
        nodes,
        live_node_count: nodes_by_kind.len(),
        reverse_index: LazyNodeIndex { nodes_by_kind, kinds },
        ..Default::default()
    }
}

#[test]
fn reverse_index_checks_full_fingerprints_and_kind_with_sparse_indices() {
    let first = node(DepKind::CompileCodegenUnit, 1, 2);
    let collision = node(DepKind::CompileCodegenUnit, 2, 1);
    let other_kind = node(DepKind::CompileMonoItem, 1, 2);
    let null = node(DepKind::Null, 0, 0);
    assert_eq!(
        fingerprint_hash(first.key_fingerprint),
        fingerprint_hash(collision.key_fingerprint)
    );
    let graph = graph(vec![null, first, null, other_kind, collision]);
    assert!(graph.reverse_index.kinds.iter().all(|kind| kind.map.get().is_none()));
    assert_eq!(graph.node_to_index_opt(&first), Some(SerializedDepNodeIndex::from_u32(1)));
    assert!(graph.reverse_index.kinds[other_kind.kind.as_usize()].map.get().is_none());
    for (node, index) in [(first, 1), (collision, 4), (other_kind, 3)] {
        assert_eq!(graph.node_to_index_opt(&node), Some(SerializedDepNodeIndex::from_u32(index)));
    }
    assert_eq!(graph.node_to_index_opt(&node(DepKind::CompileCodegenUnit, 3, 0)), None);
    assert_eq!(graph.node_to_index_opt(&node(DepKind::SideEffect, 1, 2)), None);
    assert_eq!(graph.node_to_index_opt(&null), None);
    assert_eq!(SerializedDepGraph::default().node_to_index_opt(&first), None);
}

#[test]
fn reverse_index_side_effect_duplicates_use_the_last_index() {
    let duplicate = node(DepKind::SideEffect, 1, 2);
    let collision = node(DepKind::SideEffect, 2, 1);
    let graph = graph(vec![duplicate, collision, node(DepKind::Null, 0, 0), duplicate]);
    assert_eq!(graph.node_to_index_opt(&duplicate), Some(SerializedDepNodeIndex::from_u32(3)));
    assert_eq!(graph.node_to_index_opt(&collision), Some(SerializedDepNodeIndex::from_u32(1)));
}

#[test]
#[should_panic(expected = "does not have an unique index")]
fn reverse_index_rejects_other_duplicate_nodes() {
    let duplicate = node(DepKind::CompileCodegenUnit, 1, 2);
    graph(vec![duplicate, duplicate]).node_to_index_opt(&duplicate);
}
