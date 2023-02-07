// run-pass
// compile-flags: --target x86_64-unknown-linux-gnu -Copt-level=0 -Cllvm-args=-opaque-pointers=0
// needs-llvm-components: x86

// (opaque-pointers flag is called force-opaque-pointers in LLVM 13...)
// min-llvm-version: 14.0

// This test can be removed once non-opaque pointers are gone from LLVM, maybe.

#![feature(dyn_star, pointer_sized_trait)]
#![allow(incomplete_features)]

use std::fmt::Debug;
use std::marker::PointerSized;

fn make_dyn_star<'a>(t: impl PointerSized + Debug + 'a) -> dyn* Debug + 'a {
    t as _
}

fn main() {
    println!("{:?}", make_dyn_star(Box::new(1i32)));
    println!("{:?}", make_dyn_star(2usize));
    println!("{:?}", make_dyn_star((3usize,)));
}
