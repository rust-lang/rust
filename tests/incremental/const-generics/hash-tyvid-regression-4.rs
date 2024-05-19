//@ revisions: cfail
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
// regression test for #79251
#[derive(Debug)]
struct Node<K, const D: usize>
where
    SmallVec<K, { D * 2 }>: ,
{
    keys: SmallVec<K, { D * 2 }>,
}

impl<K, const D: usize> Node<K, D>
where
    SmallVec<K, { D * 2 }>: ,
{
    fn new() -> Self {
        panic!()
    }

    #[inline(never)]
    fn split(&mut self, i: usize, k: K, right: bool) -> Node<K, D> {
        let mut node = Node::new();
        node.keys.push(k);
        //~^ error: no method named
        node
    }
}

#[derive(Debug)]
struct SmallVec<T, const D: usize> {
    data: [T; D],
}
impl<T, const D: usize> SmallVec<T, D> {
    fn new() -> Self {
        panic!()
    }
}

fn main() {}
