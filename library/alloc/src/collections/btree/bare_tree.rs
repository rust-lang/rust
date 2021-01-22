use super::node::{marker, NodeRef};

/// Simple tree around a raw `NodeRef`, that provides a destructor.
/// Unlike smarter sister `BTreeMap`, it does not keep track of element count,
/// does not allow replacing the root, therefore can be statically tyoed by the
/// node type of the root, and cannot represent a tree without root node
/// (its `Option` exists only to disable the `drop` method when needed).
pub struct BareTree<K, V, RootType>(Option<NodeRef<marker::Owned, K, V, RootType>>);

impl<K, V, RootType> Drop for BareTree<K, V, RootType> {
    fn drop(&mut self) {
        if let Some(root) = self.0.take() {
            let mut cur_edge = root.forget_type().into_dying().first_leaf_edge();
            while let Some((next_edge, kv)) = unsafe { cur_edge.deallocating_next() } {
                unsafe { kv.drop_key_val() };
                cur_edge = next_edge;
            }
        }
    }
}

impl<K, V> BareTree<K, V, marker::Leaf> {
    /// Returns a new tree consisting of a single leaf that is initially empty.
    pub fn new_leaf() -> Self {
        Self(Some(NodeRef::new_leaf()))
    }
}

impl<K, V> BareTree<K, V, marker::Internal> {
    /// Returns a new tree with an internal root, that initially has no elements
    /// and one child.
    pub fn new_internal(child: NodeRef<marker::Owned, K, V, marker::LeafOrInternal>) -> Self {
        Self(Some(NodeRef::new_internal(child)))
    }
}

impl<K, V, RootType> BareTree<K, V, RootType> {
    /// Mutably borrows the owned root node.
    pub fn borrow_mut(&mut self) -> NodeRef<marker::Mut<'_>, K, V, RootType> {
        self.0.as_mut().unwrap().borrow_mut()
    }

    /// Removes any static information asserting that the root is a `Leaf` or
    /// `Internal` node.
    pub fn forget_root_type(mut self) -> BareTree<K, V, marker::LeafOrInternal> {
        BareTree(self.0.take().map(NodeRef::forget_type))
    }

    /// Consumes the tree, returning a `NodeRef` that still enforces ownership
    /// but lacks a destructor.
    pub fn into_inner(mut self) -> NodeRef<marker::Owned, K, V, RootType> {
        self.0.take().unwrap()
    }
}
