//@ known-bug: #124350

struct Node<const D: usize> {}

impl Node<D>
where
    SmallVec<{ D * 2 }>:,
{
    fn new() -> Self {
        let mut node = Node::new();
        (&a, 0)();

        node
    }
}

struct SmallVec<T1, T2> {}
