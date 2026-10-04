//@ compile-flags: -Copt-level=3
//

#![crate_type = "lib"]

use std::iter;

// CHECK-LABEL: @repeat_take_collect
#[no_mangle]
pub fn repeat_take_collect() -> Vec<u8> {
    // CHECK: call void @llvm.memset.{{.+}}(ptr {{.*}}align 1{{.*}} %{{.*}}, i8 42, i{{[0-9]+}} 100000, i1 false)
    iter::repeat(42).take(100000).collect()
}

// CHECK-LABEL: @repeat_with_take_collect
#[no_mangle]
pub fn repeat_with_take_collect() -> Vec<u8> {
    // CHECK: call void @llvm.memset.{{.+}}(ptr {{.*}}align 1{{.*}} %{{.*}}, i8 13, i{{[0-9]+}} 12345, i1 false)
    iter::repeat_with(|| 13).take(12345).collect()
}
