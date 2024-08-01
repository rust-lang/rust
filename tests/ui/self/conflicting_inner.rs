//@ run-pass
//@ revisions: default feature
#![cfg_attr(feature, feature(arbitrary_self_types))]

// This test aims to be like the IndexVec within rustc, and conflicts
// over its into_iter().

#[allow(dead_code)]
trait Foo {
    fn foo(self) -> usize;
}

struct IndexVec<T>(T);

impl<T> std::ops::Deref for IndexVec<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, T> Foo for &'a IndexVec<T> {
    fn foo(self) -> usize {
        2
    }
}

impl<T> IndexVec<T> {
    fn foo(self) -> usize {
        1
    }
}

fn main() {
    let ivec = IndexVec(0usize);
    assert_eq!(ivec.foo(), 1);
}
