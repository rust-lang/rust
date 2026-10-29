//@ run-rustfix
//@ aux-build:external_unsafe_macro.rs

#![deny(unsafe_op_in_unsafe_fn)] //~ NOTE
#![crate_name = "wrapping_unsafe_block_sugg"]

extern crate external_unsafe_macro;

unsafe fn unsf() {}

pub unsafe fn foo() {
    //~^ NOTE an unsafe function restricts its caller, but its body is safe by default
    unsf(); //~ ERROR call to unsafe function `unsf` is unsafe
    //~^ NOTE call to unsafe function
    //~| NOTE for more information, see
    //~| NOTE consult the function's documentation
    unsf(); //~ ERROR call to unsafe function `unsf` is unsafe
    //~^ NOTE call to unsafe function
    //~| NOTE for more information, see
    //~| NOTE consult the function's documentation
}

pub unsafe fn bar(x: *const i32) -> i32 {
    //~^ NOTE an unsafe function restricts its caller, but its body is safe by default
    let y = *x; //~ ERROR dereference of raw pointer is unsafe and requires unsafe block
    //~^ NOTE dereference of raw pointer
    //~| NOTE for more information, see
    //~| NOTE raw pointers may be null
    y + *x //~ ERROR dereference of raw pointer is unsafe and requires unsafe block
    //~^ NOTE dereference of raw pointer
    //~| NOTE for more information, see
    //~| NOTE raw pointers may be null
}

static mut BAZ: i32 = 0;
pub unsafe fn baz() -> i32 {
    //~^ NOTE an unsafe function restricts its caller, but its body is safe by default
    let y = BAZ; //~ ERROR use of mutable static is unsafe and requires unsafe block
    //~^ NOTE use of mutable static
    //~| NOTE for more information, see
    //~| NOTE mutable statics can be mutated by multiple threads
    y + BAZ //~ ERROR use of mutable static is unsafe and requires unsafe block
    //~^ NOTE use of mutable static
    //~| NOTE for more information, see
    //~| NOTE mutable statics can be mutated by multiple threads
}

macro_rules! unsafe_macro { () => (unsf()) }
//~^ ERROR call to unsafe function `unsf` is unsafe
//~| NOTE call to unsafe function
//~| NOTE for more information, see
//~| NOTE consult the function's documentation
//~| ERROR call to unsafe function `unsf` is unsafe
//~| NOTE call to unsafe function
//~| NOTE for more information, see
//~| NOTE consult the function's documentation

pub unsafe fn unsafe_in_macro() {
    //~^ NOTE an unsafe function restricts its caller, but its body is safe by default
    unsafe_macro!();
    //~^ NOTE in this expansion
    //~| NOTE in this expansion
    //~| NOTE in this expansion
    unsafe_macro!();
    //~^ NOTE in this expansion
    //~| NOTE in this expansion
    //~| NOTE in this expansion
}

pub unsafe fn unsafe_in_external_macro() {
    // FIXME: https://github.com/rust-lang/rust/issues/112504
    // FIXME: ~^ NOTE an unsafe function restricts its caller, but its body is safe by default
    external_unsafe_macro::unsafe_macro!();
    external_unsafe_macro::unsafe_macro!();
}

fn main() {}
