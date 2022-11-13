// compile-flags: -O

#![crate_type = "lib"]

// CHECK-LABEL: @none_repeat
#[no_mangle]
pub fn none_repeat() -> [Option<u8>; 64] {
    // CHECK: store <128 x i8>
    // CHECK-NEXT: ret void
    [None; 64]
}

// CHECK-LABEL: @some_repeat
#[no_mangle]
pub fn some_repeat() -> [Option<u8>; 64] {
    // CHECK: store <128 x i8>
    // CHECK-NEXT: ret void
    [Some(0); 64]
}
