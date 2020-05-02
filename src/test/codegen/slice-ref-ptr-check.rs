// ignore-tidy-linelength
// compile-flags: -C opt-level=3

#![crate_type = "lib"]

// https://github.com/rust-lang/rust/issues/71602

// CHECK-LABEL: @is_zero_slice
#[no_mangle]
pub fn is_zero_slice(data: &[u8; 4]) -> bool {
    // CHECK: start
    // CHECK-NOT: %_8.i.i.i = icmp eq [4 x i8]* %data, getelementptr inbounds (<{ [4 x i8] }>, <{ [4 x i8] }>* @alloc2, i64 0, i32 0)
    // CHECK: ret
    *data == [0; 4]
}
