// pretty-compare-only
// pretty-mode:hir
// pp-exact:hir-fn-variadic.pp

#![feature(c_variadic)]
#[prelude_import]
use ::std::prelude::rust_2015::*;
#[macro_use]
extern crate std;

extern "C" {
    fn foo(x: i32, va1: ...);
}

unsafe extern "C" fn bar(_: i32, mut va2: ...) -> usize { va2.arg::<usize>() }
