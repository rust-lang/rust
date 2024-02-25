//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-fn-variadic.pp

#![feature(c_variadic)]

extern "C" {
    pub fn foo(x: i32, va1: ...);
}

pub unsafe extern "C" fn bar(_: i32, mut va2: ...) -> usize {
    va2.arg::<usize>()
}
