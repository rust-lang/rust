//@ check-pass

#![feature(fn_delegation)]

pub trait Idx {
    fn new(idx: usize) -> Self;

    fn index(self) -> usize;
}

struct DefIndex(usize);
impl Idx for DefIndex {
    fn new(idx: usize) -> Self { DefIndex(idx) }
    fn index(self) -> usize { self.0 }
}

struct LocalDefId {
    index: DefIndex
}

impl Idx for LocalDefId {
    reuse <DefIndex as Idx>::new;
    reuse Idx::index { self.index }
}

fn main() {}
