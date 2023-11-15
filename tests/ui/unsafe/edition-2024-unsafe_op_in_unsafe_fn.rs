// edition: 2024
// compile-flags: -Zunstable-options
// check-pass
// revisions: mir thir
// [thir]compile-flags: -Zthir-unsafeck

#![crate_type = "lib"]
#![deny(unused_unsafe)]

unsafe fn unsf() {}

unsafe fn foo() {
    unsf();
    //[mir]~^ WARN call to unsafe function is unsafe and requires unsafe block
    //[thir]~^^ WARN call to unsafe function `unsf` is unsafe and requires unsafe block

    // no unused_unsafe
    unsafe {
        unsf();
    }
}
