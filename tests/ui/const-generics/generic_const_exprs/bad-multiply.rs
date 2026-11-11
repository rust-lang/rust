// regression test for #124350

struct Node<const D: usize> {}

impl<const D: usize> Node<D>
where
    SmallVec<{ D * 2 }>:,
    //~^ ERROR generic parameters may not be used in const operations
    //~| ERROR constant provided when a type was expected
{
    fn new() -> Self {
        Node::new()
    }
}

struct SmallVec<T1>(T1);

fn main() {}
