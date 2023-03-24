// compile-flags: -C no-prepopulate-passes -Copt-level=0

#![crate_type = "lib"]

// Hack to get the correct size for the length part in slices
// CHECK: @helper([[USIZE:i[0-9]+]] %_1)
#[no_mangle]
pub fn helper(_: usize) {
}

// CHECK-LABEL: @no_op_slice_adjustment
#[no_mangle]
pub fn no_op_slice_adjustment(x: &[u8]) -> &[u8] {
    // We used to generate an extra alloca and memcpy for the block's trailing expression value, so
    // check that we copy directly to the return value slot
// CHECK: %0 = insertvalue { {{\[0 x i8\]\*|ptr}}, [[USIZE]] } poison, {{\[0 x i8\]\*|ptr}} %x.0, 0
// CHECK: %1 = insertvalue { {{\[0 x i8\]\*|ptr}}, [[USIZE]] } %0, [[USIZE]] %x.1, 1
// CHECK: ret { {{\[0 x i8\]\*|ptr}}, [[USIZE]] } %1
    { x }
}

// CHECK-LABEL: @no_op_slice_adjustment2
#[no_mangle]
pub fn no_op_slice_adjustment2(x: &[u8]) -> &[u8] {
    // We used to generate an extra alloca and memcpy for the function's return value, so check
    // that there's no memcpy (the slice is written to sret_slot element-wise)
// CHECK-NOT: call void @llvm.memcpy.
    no_op_slice_adjustment(x)
}
