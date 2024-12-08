//@ revisions: cfail
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
// regression test for #79251
struct Node<const D: usize>
where
    SmallVec<{ D * 2 }>: ,
{
    keys: SmallVec<{ D * 2 }>,
}

impl<const D: usize> Node<D>
where
    SmallVec<{ D * 2 }>: ,
{
    fn new() -> Self {
        let mut node = Node::new();
        node.keys.some_function();
        //~^ error: no method named
        node
    }
}

struct SmallVec<const D: usize> {}

fn main() {}
