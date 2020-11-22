use core::borrow::Borrow;
use core::cmp::Ordering;

use super::node::{marker, ForceResult::*, Handle, NodeRef};

use SearchResult::*;

pub enum SearchResult<BorrowType, K, V, FoundType, GoDownType> {
    Found(Handle<NodeRef<BorrowType, K, V, FoundType>, marker::KV>),
    GoDown(Handle<NodeRef<BorrowType, K, V, GoDownType>, marker::Edge>),
}

/// Looks up a given key in a (sub)tree headed by the given node, recursively.
/// Returns a `Found` with the handle of the matching KV, if any. Otherwise,
/// returns a `GoDown` with the handle of the possible leaf edge where the key
/// belongs.
pub fn search_tree<BorrowType, K, V, Q: ?Sized>(
    mut node: NodeRef<BorrowType, K, V, marker::LeafOrInternal>,
    key: &Q,
) -> SearchResult<BorrowType, K, V, marker::LeafOrInternal, marker::Leaf>
where
    Q: Ord,
    K: Borrow<Q>,
{
    loop {
        match search_node(node, key) {
            Found(handle) => return Found(handle),
            GoDown(handle) => match handle.force() {
                Leaf(leaf) => return GoDown(leaf),
                Internal(internal) => {
                    node = internal.descend();
                    continue;
                }
            },
        }
    }
}

/// Looks up a given key in a given node, without recursion.
/// Returns a `Found` with the handle of the matching KV, if any. Otherwise,
/// returns a `GoDown` with the handle of the edge where the key might be found.
/// If the node is a leaf, a `GoDown` edge is not an actual edge but a possible edge.
pub fn search_node<BorrowType, K, V, Type, Q: ?Sized>(
    node: NodeRef<BorrowType, K, V, Type>,
    key: &Q,
) -> SearchResult<BorrowType, K, V, Type, Type>
where
    Q: Ord,
    K: Borrow<Q>,
{
    match search_linear(&node, key) {
        (idx, true) => Found(unsafe { Handle::new_kv(node, idx) }),
        (idx, false) => GoDown(unsafe { Handle::new_edge(node, idx) }),
    }
}

/// Returns the index in the node at which the key (or an equivalent) exists
/// or could exist, and whether it exists in the node itself. If it doesn't
/// exist in the node itself, it may exist in the subtree with that index
/// (if the node has subtrees). If the key doesn't exist in node or subtree,
/// the returned index is the position or subtree where the key belongs.
fn search_linear<BorrowType, K, V, Type, Q: ?Sized>(
    node: &NodeRef<BorrowType, K, V, Type>,
    key: &Q,
) -> (usize, bool)
where
    Q: Ord,
    K: Borrow<Q>,
{
    // This function is defined over all borrow types (immutable, mutable, owned).
    // Using `keys_at()` is fine here even if BorrowType is mutable, as all we return
    // is an index -- not a reference.
    let len = node.len();
    for i in 0..len {
        let k = unsafe { node.reborrow().key_at(i) };
        match key.cmp(k.borrow()) {
            Ordering::Greater => {}
            Ordering::Equal => return (i, true),
            Ordering::Less => return (i, false),
        }
    }
    (len, false)
}
