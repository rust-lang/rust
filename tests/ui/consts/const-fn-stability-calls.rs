//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
// Test use of const fn from another crate without a feature gate.

//@ aux-build:const_fn_lib.rs

extern crate const_fn_lib;

use const_fn_lib::foo;

static FOO: usize = foo();
const BAR: usize = foo();

macro_rules! constant {
    ($n:ident: $t:ty = $v:expr) => {
        const $n: $t = $v;
    }
}

constant! {
    BAZ: usize = foo()
}

fn main() {
    let x: [usize; foo()] = [42; foo()];
}
