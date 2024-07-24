struct Node<const D: usize> {}

impl<const D: usize> Node<{ D }>
where
    SmallVec<D>:, //~ ERROR: constant provided when a type was expected
{
    fn new() {}
}

struct SmallVec<T>(T);

fn main() {
    Node::new();
}
