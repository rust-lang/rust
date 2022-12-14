// This test checks that temporaries for indirectly-passed arguments get lifetime markers.

// compile-flags: -O -C no-prepopulate-passes -Zmir-opt-level=0

#![crate_type = "lib"]

extern "Rust" {
    fn f(x: [u8; 1024]);
}

const A: [u8; 1024] = [0; 1024];

// CHECK-LABEL: @const_arg_indirect
#[no_mangle]
pub unsafe fn const_arg_indirect() {
    // Ensure that the live ranges for the two argument temporaries don't overlap.

    // CHECK: call void @llvm.lifetime.start
    // CHECK: call void @f
    // CHECK: call void @llvm.lifetime.end
    // CHECK: call void @llvm.lifetime.start
    // CHECK: call void @f
    // CHECK: call void @llvm.lifetime.end

    f(A);
    f(A);
}
