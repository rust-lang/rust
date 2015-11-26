// Test impls

impl<T> JSTraceable for SmallVec<[T; 1]> {}

impl<K, V, NodeRef: Deref<Target = Node<K, V>>> Handle<NodeRef, handle::Edge, handle::Internal> {
    // Keep this.
}
