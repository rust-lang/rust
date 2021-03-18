// run-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

pub trait Foo {
    fn foo(&self);
}


impl<T, const N: usize> Foo for [T; N] {
    fn foo(&self) {
        let _ = &self;
    }
}

fn main() {}
