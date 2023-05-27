// run-rustfix

#![deny(unsafe_op_in_unsafe_fn)] //~ NOTE

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

fn main() {}
