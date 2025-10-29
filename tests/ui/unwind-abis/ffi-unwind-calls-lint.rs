//@ build-pass
//@ needs-unwind

#![feature(rustc_attrs)]
#![warn(ffi_unwind_calls)]

mod foo {
    #[no_mangle]
    pub extern "C-unwind" fn foo() {}
}

extern "C-unwind" {
    fn foo();
}

fn main() {
    // Call to Rust function is fine.
    foo::foo();
    // Call to foreign function should warn.
    unsafe {
        foo();
    }
    //~^ WARNING call to foreign function with FFI-unwind ABI
    let ptr: extern "C-unwind" fn() = foo::foo;
    // Call to function pointer should also warn.
    ptr();
    //~^ WARNING call to function pointer with FFI-unwind ABI
}

#[rustc_propagate_ffi_unwind]
fn f() {
    // Call to foreign function or a function pointer from within a `#[rustc_propagate_ffi_unwind]`
    // function is fine.
    unsafe {
        foo();
    }
    let ptr: extern "C-unwind" fn() = foo::foo;
    ptr();
}
