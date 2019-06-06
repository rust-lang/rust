// compile-flags: -O
// ignore-tidy-linelength
// min-llvm-version 7.0

#![crate_type = "lib"]

use std::iter;

// CHECK: @helper([[USIZE:i[0-9]+]] %_1)
#[no_mangle]
pub fn helper(_: usize) {
}

// CHECK-LABEL: @repeat_take_collect
#[no_mangle]
pub fn repeat_take_collect() -> Vec<u8> {
// FIXME: At the time of writing LLVM transforms this loop into a single
// `store` and then a `memset` with size = 99999. The correct check should be:
//        call void @llvm.memset.p0i8.[[USIZE]](i8* {{(nonnull )?}}align 1 %{{[a-z0-9.]+}}, i8 42, [[USIZE]] 100000, i1 false)

// CHECK: call void @llvm.memset.p0i8.[[USIZE]](i8* {{(nonnull )?}}align 1 %{{[a-z0-9.]+}}, i8 42, [[USIZE]] 99999, i1 false)
    iter::repeat(42).take(100000).collect()
}
