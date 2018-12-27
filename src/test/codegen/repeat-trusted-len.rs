// compile-flags: -O
// ignore-tidy-linelength
// min-llvm-version 7.0

#![crate_type = "lib"]

use std::iter;

// CHECK: @helper([[USIZE:i[0-9]+]] %arg0)
#[no_mangle]
pub fn helper(_: usize) {
}

// CHECK-LABEL: @repeat_take_collect
#[no_mangle]
pub fn repeat_take_collect() -> Vec<u8> {
// CHECK: call void @llvm.memset.p0i8.[[USIZE]](i8* {{(nonnull )?}}align 1 %{{[0-9]+}}, i8 42, [[USIZE]] 100000, i1 false)
    iter::repeat(42).take(100000).collect()
}
