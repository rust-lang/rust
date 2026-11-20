use super::*;

fn node(kind: DepKind, first: u64, second: u64) -> DepNode {
    DepNode { kind, key_fingerprint: Fingerprint::new(first, second).into() }
}

fn single_kind_graph(nodes: Vec<DepNode>) -> SerializedDepGraph {
    let kind = nodes[0].kind;
    let nodes = IndexVec::from_raw(nodes);
    let mut kinds = Vec::new();
    kinds.resize_with(DepKind::NUM_VARIANTS, LazyKindIndex::default);
    kinds[kind.as_usize()].len = nodes.len() as u32;
    SerializedDepGraph {
        live_node_count: nodes.len(),
        reverse_index: LazyNodeIndex { nodes_by_kind: nodes.indices().map(Some).collect(), kinds },
        nodes,
        ..Default::default()
    }
}

#[test]
fn reverse_index_distinguishes_hash_collisions() {
    let first = node(DepKind::CompileCodegenUnit, 1, 2);
    let collision = node(DepKind::CompileCodegenUnit, 2, 1);
    assert_eq!(
        fingerprint_hash(first.key_fingerprint),
        fingerprint_hash(collision.key_fingerprint)
    );
    let graph = single_kind_graph(vec![first, collision]);
    assert_eq!(graph.node_to_index_opt(&first), Some(SerializedDepNodeIndex::from_u32(0)));
    assert_eq!(graph.node_to_index_opt(&collision), Some(SerializedDepNodeIndex::from_u32(1)));
    assert_eq!(graph.node_to_index_opt(&node(DepKind::CompileCodegenUnit, 3, 0)), None);
}

#[test]
fn reverse_index_side_effect_duplicates_use_the_last_index() {
    let duplicate = node(DepKind::SideEffect, 1, 2);
    let graph = single_kind_graph(vec![duplicate, duplicate]);
    assert_eq!(graph.node_to_index_opt(&duplicate), Some(SerializedDepNodeIndex::from_u32(1)));
}

#[test]
#[should_panic(expected = "does not have an unique index")]
fn reverse_index_rejects_other_duplicate_nodes() {
    let duplicate = node(DepKind::CompileCodegenUnit, 1, 2);
    single_kind_graph(vec![duplicate, duplicate]).node_to_index_opt(&duplicate);
}
