// Simple test that checks if the writeable flag works.
//@compile-flags: -Zmiri-tree-borrows

#![feature(rustc_attrs)]
#![allow(internal_features)]

fn main() {
    let mut x = 1;
    foo(&mut x);
    assert_eq!(x, 2);

    let bar = Bar {};
    bar.foo(&mut x);
    assert_eq!(x, 3);
}

#[rustc_no_writable]
fn foo(x: &mut i32) {
    *x += 1;
}

struct Bar {}

impl Bar {
    #[rustc_no_writable]
    fn foo(&self, x: &mut i32) {
        crate::foo(x)
    }
}
