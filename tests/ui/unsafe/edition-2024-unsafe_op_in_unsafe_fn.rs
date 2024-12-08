//@ edition: 2024
//@ check-pass
#![crate_type = "lib"]
#![deny(unused_unsafe)]

unsafe fn unsf() {}

unsafe fn foo() {
    unsf();
    //~^ WARN call to unsafe function `unsf` is unsafe and requires unsafe block

    // no unused_unsafe
    unsafe {
        unsf();
    }
}
