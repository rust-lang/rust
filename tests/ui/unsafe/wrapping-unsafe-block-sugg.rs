// run-rustfix
// aux-build:external_unsafe_macro.rs

#![deny(unsafe_op_in_unsafe_fn)] //~ NOTE

extern crate external_unsafe_macro;

unsafe fn unsf() {}

pub unsafe fn foo() {
    //~^ NOTE an unsafe function restricts its caller, but its body is safe by default
    unsf(); //~ ERROR call to unsafe function is unsafe
    //~^ NOTE
    //~| NOTE
    unsf(); //~ ERROR call to unsafe function is unsafe
    //~^ NOTE
    //~| NOTE
}

pub unsafe fn bar(x: *const i32) -> i32 {
    //~^ NOTE an unsafe function restricts its caller, but its body is safe by default
    let y = *x; //~ ERROR dereference of raw pointer is unsafe and requires unsafe block
    //~^ NOTE
    //~| NOTE
    y + *x //~ ERROR dereference of raw pointer is unsafe and requires unsafe block
    //~^ NOTE
    //~| NOTE
}

static mut BAZ: i32 = 0;
pub unsafe fn baz() -> i32 {
    //~^ NOTE an unsafe function restricts its caller, but its body is safe by default
    let y = BAZ; //~ ERROR use of mutable static is unsafe and requires unsafe block
    //~^ NOTE
    //~| NOTE
    y + BAZ //~ ERROR use of mutable static is unsafe and requires unsafe block
    //~^ NOTE
    //~| NOTE
}

macro_rules! unsafe_macro { () => (unsf()) }
//~^ ERROR call to unsafe function is unsafe
//~| NOTE
//~| NOTE
//~| ERROR call to unsafe function is unsafe
//~| NOTE
//~| NOTE

pub unsafe fn unsafe_in_macro() {
    //~^ NOTE an unsafe function restricts its caller, but its body is safe by default
    unsafe_macro!();
    //~^ NOTE
    //~| NOTE
    unsafe_macro!();
    //~^ NOTE
    //~| NOTE
}

pub unsafe fn unsafe_in_external_macro() {
    // FIXME: https://github.com/rust-lang/rust/issues/112504
    // FIXME: ~^ NOTE an unsafe function restricts its caller, but its body is safe by default
    external_unsafe_macro::unsafe_macro!();
    external_unsafe_macro::unsafe_macro!();
}

fn main() {}
