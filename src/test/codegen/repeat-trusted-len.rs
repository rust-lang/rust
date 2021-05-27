// compile-flags: -O
//

#![crate_type = "lib"]

use std::iter;

// CHECK-LABEL: @repeat_take_collect
#[no_mangle]
pub fn repeat_take_collect() -> Vec<u8> {
// CHECK: call void @llvm.memset.p0i8.i{{[0-9]+}}(i8* {{(nonnull )?}}align 1{{.*}} %{{[0-9]+}}, i8 42, i{{[0-9]+}} 100000, i1 false)
    iter::repeat(42).take(100000).collect()
}
