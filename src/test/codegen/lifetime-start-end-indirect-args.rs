// This test checks that moved arguments passed indirectly get lifetime markers.

// compile-flags: -O -C no-prepopulate-passes -Zmir-opt-level=0

#![crate_type = "lib"]

// CHECK-LABEL: @arg_indirect
#[no_mangle]
pub fn arg_indirect(a: [u8; 1234]) {
    // Arguments passed indirectly should get lifetime markers.

    // CHECK: [[A:%[0-9]+]] = bitcast{{.*}} %a
    // CHECK: call void @llvm.lifetime.end{{.*}} [[A]])
}

// CHECK-LABEL: @arg_by_val
#[no_mangle]
pub fn arg_by_val(a: u8) {
    // Arguments passed by value should not get lifetime markers.

    // CHECK-NOT: call void @llvm.lifetime.end
}

// CHECK-LABEL: @arg_by_ref
#[no_mangle]
pub fn arg_by_ref(a: &[u8; 1234]) {
    // Arguments passed by ref should not get lifetime markers.

    // CHECK-NOT: call void @llvm.lifetime.end
}

// CHECK-LABEL: @with_other_args
#[no_mangle]
pub fn with_other_args(x: (), y: (), a: [u8; 1234], z: ()) {
    // Lifetime markers should be applied to the correct argument,
    // even in the presence of ignored ZST arguments.

    // CHECK-NOT: call void @llvm.lifetime.end
    // CHECK: [[A:%[0-9]+]] = bitcast{{.*}} %a
    // CHECK: call void @llvm.lifetime.end{{.*}} [[A]])
    // CHECK-NOT: call void @llvm.lifetime.end
}

// CHECK-LABEL: @forward_directly_to_ret
#[no_mangle]
pub fn forward_directly_to_ret(a: [u8; 1234]) -> [u8; 1234] {
    // Ensure that lifetime markers appear after the argument is copied to the return place.
    // (Since reading from `a` after `lifetime.end` would return poison.)

    // CHECK: memcpy
    // CHECK: [[A:%[0-9]+]] = bitcast{{.*}} %a
    // CHECK: call void @llvm.lifetime.end{{.*}} [[A]])
    // CHECK-NEXT: ret void
    a
}

pub struct LargeWithField {
    x: u8,
    _rest: [u8; 1234],
}

// CHECK-LABEL: @read_from_arg
#[no_mangle]
pub fn read_from_arg(a: LargeWithField) -> u8 {
    // Ensure that lifetime markers appear after reading from the argument.
    // (Since reading from `a` after `lifetime.end` would return poison.)

    // CHECK: [[LOAD:%[0-9]+]] = load i8
    // CHECK: [[A:%[0-9]+]] = bitcast{{.*}} %a
    // CHECK: call void @llvm.lifetime.end{{.*}} [[A]])
    // CHECK-NEXT: ret i8 [[LOAD]]
    a.x
}
