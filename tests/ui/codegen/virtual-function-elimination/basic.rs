//@ build-pass
//@ compile-flags: -Zvirtual-function-elimination=true -Clto=true -Zverify-llvm-ir
//@ only-x86_64
//@ no-prefer-dynamic
//@ ignore-backends: gcc
#![feature(ptr_metadata)]
#![crate_type = "staticlib"]

// issue #123955
#[no_mangle]
fn test0() {
    _ = Box::new(()) as Box<dyn Send>;
}

// issue #124092
const X: for<'b> fn(&'b ()) = |&()| ();
#[no_mangle]
fn test1() {
    let _dyn_debug = Box::new(X) as Box<fn(&'static ())> as Box<dyn Send>;
}

// issue #164267
#[no_mangle]
fn test2() {
    let x: &dyn std::fmt::Debug = &0u8;
    // Reads a non-function-pointer from the vtable.
    let _ = std::mem::align_of_val(x);
}

#[no_mangle]
fn test3() {
    let x: &dyn std::fmt::Debug = &0u8;
    std::ptr::metadata(x).size_of();
}
