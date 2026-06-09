#![deny(rust_2024_compatibility)]
#![crate_type = "lib"]

unsafe fn unsf() {}

unsafe fn foo() {
    unsf();
    //~^ ERROR call to unsafe function `unsf` is unsafe and requires unsafe block
}
