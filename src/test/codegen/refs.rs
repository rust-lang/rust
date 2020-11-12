// compile-flags: -C no-prepopulate-passes -Zmir-opt-level=0

#![crate_type = "lib"]

// Hack to get the correct size for the length part in slices
// CHECK: @helper([[USIZE:i[0-9]+]] %_1)
#[no_mangle]
pub fn helper(_: usize) {
}

// CHECK-LABEL: @ref_dst
#[no_mangle]
pub fn ref_dst(s: &[u8]) {
    // We used to generate an extra alloca and memcpy to ref the dst, so check that we copy
    // directly to the alloca for "x"
// CHECK: [[X0:%[0-9]+]] = getelementptr {{.*}} { [0 x i8]*, [[USIZE]] }* %x, i32 0, i32 0
// CHECK: store [0 x i8]* %s.0, [0 x i8]** [[X0]]
// CHECK: [[X1:%[0-9]+]] = getelementptr {{.*}} { [0 x i8]*, [[USIZE]] }* %x, i32 0, i32 1
// CHECK: store [[USIZE]] %s.1, [[USIZE]]* [[X1]]

    let x = &*s;
    &x; // keep variable in an alloca
}
