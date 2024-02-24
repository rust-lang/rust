// Regression test for #34991: an ICE occurred here because we inline
// some of the vector routines and give them a local def-id `X`. This
// got hashed after codegen (`hir_owner(X)`). When we load back up, we get an
// error because the `X` is remapped to the original def-id (in
// libstd), and we can't hash a HIR node from std.

//@ revisions:rpass1 rpass2

#![feature(rustc_attrs)]

use std::vec::Vec;

pub fn foo() -> Vec<i32> {
    vec![1, 2, 3]
}

pub fn bar() {
    foo();
}

pub fn main() {
    bar();
}
