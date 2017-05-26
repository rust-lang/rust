// Test impls

impl<T> JSTraceable for SmallVec<[T; 1]> {}

impl<K, V, NodeRef: Deref<Target = Node<K, V>>> Handle<NodeRef, handle::Edge, handle::Internal> {
    // Keep this.
}

impl<V> Test<V>
    where V: Clone // This comment is NOT removed by formating!
{
    pub fn new(value: V) -> Self {
        Test {
            cloned_value: value.clone(),
            value: value,
        }
    }
}
