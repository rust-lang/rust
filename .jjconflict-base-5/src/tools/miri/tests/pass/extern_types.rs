//@revisions: stack tree tree_implicit_writes
//@[tree_implicit_writes]compile-flags: -Zmiri-tree-borrows -Zmiri-tree-borrows-implicit-writes
//@[tree]compile-flags: -Zmiri-tree-borrows
#![feature(extern_types)]

use std::ptr;

extern "C" {
    type Foo;
}

fn main() {
    let x: &Foo = unsafe { &*(ptr::without_provenance::<()>(16) as *const Foo) };
    let _y: &Foo = &*x;
}
