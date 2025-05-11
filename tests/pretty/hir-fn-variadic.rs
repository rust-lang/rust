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

fn main() {
    fn g1(_: extern "C" fn(_: u8, va: ...)) {}
    fn g2(_: extern "C" fn(_: u8, ...)) {}
    fn g3(_: extern "C" fn(u8, va: ...)) {}
    fn g4(_: extern "C" fn(u8, ...)) {}

    fn g5(_: extern "C" fn(va: ...)) {}
    fn g6(_: extern "C" fn(...)) {}

    _ = { unsafe extern "C" fn f1(_: u8, va: ...) {} };
    _ = { unsafe extern "C" fn f2(_: u8, ...) {} };

    _ = { unsafe extern "C" fn f5(va: ...) {} };
    _ = { unsafe extern "C" fn f6(...) {} };
}
