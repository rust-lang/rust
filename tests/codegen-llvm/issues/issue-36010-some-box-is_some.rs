#![crate_type = "lib"]

//@ compile-flags: -Copt-level=3

use std::mem;

fn foo<T>(a: &mut T, b: T) -> bool {
    let b = Some(mem::replace(a, b));
    let ret = b.is_some();
    mem::forget(b);
    return ret;
}

// CHECK-LABEL: @foo_u32
// CHECK: store i32
// CHECK-NEXT: ret i1 true
#[no_mangle]
pub fn foo_u32(a: &mut u32, b: u32) -> bool {
    foo(a, b)
}

// CHECK-LABEL: @foo_box
// CHECK: store ptr
// CHECK-NEXT: ret i1 true
#[no_mangle]
pub fn foo_box(a: &mut Box<u32>, b: Box<u32>) -> bool {
    foo(a, b)
}
