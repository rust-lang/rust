use super::{
    RecursiveEvidenceContext, RecursiveEvidenceShape, recursive_context_is_allowed,
    recursive_leaf_shape_is_allowed,
};

#[test]
fn recursive_leaf_requires_a_nested_or_serialized_position() {
    assert!(!recursive_context_is_allowed(RecursiveEvidenceContext::Complete));
    assert!(recursive_context_is_allowed(RecursiveEvidenceContext::SerializedLeaf));
    assert!(recursive_context_is_allowed(RecursiveEvidenceContext::Nested));
}

#[test]
fn recursive_leaf_must_be_an_exact_singleton() {
    let exact_leaf = RecursiveEvidenceShape {
        node_count: 1,
        root: 0,
        recursive_node: 0,
        same_scope_children: 0,
        nested_query_children: 0,
    };
    assert!(recursive_leaf_shape_is_allowed(RecursiveEvidenceContext::SerializedLeaf, exact_leaf,));
    assert!(!recursive_leaf_shape_is_allowed(RecursiveEvidenceContext::Complete, exact_leaf,));
    assert!(!recursive_leaf_shape_is_allowed(
        RecursiveEvidenceContext::SerializedLeaf,
        RecursiveEvidenceShape { node_count: 2, ..exact_leaf },
    ));
    assert!(!recursive_leaf_shape_is_allowed(
        RecursiveEvidenceContext::SerializedLeaf,
        RecursiveEvidenceShape { root: 1, ..exact_leaf },
    ));
    assert!(!recursive_leaf_shape_is_allowed(
        RecursiveEvidenceContext::SerializedLeaf,
        RecursiveEvidenceShape { same_scope_children: 1, ..exact_leaf },
    ));
    assert!(!recursive_leaf_shape_is_allowed(
        RecursiveEvidenceContext::SerializedLeaf,
        RecursiveEvidenceShape { nested_query_children: 1, ..exact_leaf },
    ));
}
