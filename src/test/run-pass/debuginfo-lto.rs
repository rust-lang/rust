// This test case makes sure that we don't run into LLVM's dreaded
// "possible ODR violation" assertion when compiling with LTO + Debuginfo.
// It covers cases that have traditionally been prone to cause this error.
// If new cases emerge, add them to this file.

// aux-build:debuginfo-lto-aux.rs
// compile-flags: -C lto -g
// no-prefer-dynamic

extern crate debuginfo_lto_aux;

fn some_fn(x: i32) -> i32 {
    x + 1
}

fn main() {
    let i = 0;
    let _ = debuginfo_lto_aux::mk_struct_with_lt(&i);
    let _ = debuginfo_lto_aux::mk_regular_struct(1);
    let _ = debuginfo_lto_aux::take_fn(some_fn, 1);
    let _ = debuginfo_lto_aux::with_closure(22);
    let _ = debuginfo_lto_aux::generic_fn(0f32);
}
