// run-rustfix

#![deny(unsafe_op_in_unsafe_fn)]

unsafe fn unsf() {}

pub unsafe fn foo() {
    unsf(); //~ ERROR call to unsafe function is unsafe
    unsf(); //~ ERROR call to unsafe function is unsafe
}

pub unsafe fn bar(x: *const i32) -> i32 {
    let y = *x; //~ ERROR dereference of raw pointer is unsafe and requires unsafe block
    y + *x //~ ERROR dereference of raw pointer is unsafe and requires unsafe block
}

static mut BAZ: i32 = 0;
pub unsafe fn baz() -> i32 {
    let y = BAZ; //~ ERROR use of mutable static is unsafe and requires unsafe block
    y + BAZ //~ ERROR use of mutable static is unsafe and requires unsafe block
}

fn main() {}
