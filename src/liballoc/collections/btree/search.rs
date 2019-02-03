use core::borrow::Borrow;
use core::cmp::Ordering;

use super::node::{Handle, NodeRef, marker, ForceResult::*};

use SearchResult::*;

pub enum SearchResult<BorrowType, K, V, FoundType, GoDownType> {
    Found(Handle<NodeRef<BorrowType, K, V, FoundType>, marker::KV>),
    GoDown(Handle<NodeRef<BorrowType, K, V, GoDownType>, marker::Edge>)
}

pub fn search_tree<BorrowType, K, V, Q: ?Sized>(
    mut node: NodeRef<BorrowType, K, V, marker::LeafOrInternal>,
    key: &Q
) -> SearchResult<BorrowType, K, V, marker::LeafOrInternal, marker::Leaf>
        where Q: Ord, K: Borrow<Q> {

    loop {
        match search_node(node, key) {
            Found(handle) => return Found(handle),
            GoDown(handle) => match handle.force() {
                Leaf(leaf) => return GoDown(leaf),
                Internal(internal) => {
                    node = internal.descend();
                    continue;
                }
            }
        }
    }
}

pub fn search_node<BorrowType, K, V, Type, Q: ?Sized>(
    node: NodeRef<BorrowType, K, V, Type>,
    key: &Q
) -> SearchResult<BorrowType, K, V, Type, Type>
        where Q: Ord, K: Borrow<Q> {

    match search_linear(&node, key) {
        (idx, true) => Found(
            Handle::new_kv(node, idx)
        ),
        (idx, false) => SearchResult::GoDown(
            Handle::new_edge(node, idx)
        )
    }
}

pub fn search_linear<BorrowType, K, V, Type, Q: ?Sized>(
    node: &NodeRef<BorrowType, K, V, Type>,
    key: &Q
) -> (usize, bool)
        where Q: Ord, K: Borrow<Q> {

    for (i, k) in node.keys().iter().enumerate() {
        match key.cmp(k.borrow()) {
            Ordering::Greater => {},
            Ordering::Equal => return (i, true),
            Ordering::Less => return (i, false)
        }
    }
    (node.keys().len(), false)
}
