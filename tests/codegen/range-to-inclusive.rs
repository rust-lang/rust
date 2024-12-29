//@ compile-flags: -O

#![crate_type = "lib"]

// CHECK-LABEL: @and_stuff
#[no_mangle]
pub fn and_stuff(a: i32, mut b: i32) -> i32 {
    // CHECK: start:
    // CHECK-NEXT: and
    // CHECK-NEXT: ret
    for _ in 0..=64 {
        b &= a;
    }

    b
}
