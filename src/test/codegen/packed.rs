//
// compile-flags: -O -C no-prepopulate-passes

#![crate_type = "lib"]

#[repr(packed)]
pub struct Packed1 {
    dealign: u8,
    data: u32
}

#[repr(packed(2))]
pub struct Packed2 {
    dealign: u8,
    data: u32
}

// CHECK-LABEL: @write_pkd1
#[no_mangle]
pub fn write_pkd1(pkd: &mut Packed1) -> u32 {
// CHECK: %{{.*}} = load i32, i32* %{{.*}}, align 1
// CHECK: store i32 42, i32* %{{.*}}, align 1
    let result = pkd.data;
    pkd.data = 42;
    result
}

// CHECK-LABEL: @write_pkd2
#[no_mangle]
pub fn write_pkd2(pkd: &mut Packed2) -> u32 {
// CHECK: %{{.*}} = load i32, i32* %{{.*}}, align 2
// CHECK: store i32 42, i32* %{{.*}}, align 2
    let result = pkd.data;
    pkd.data = 42;
    result
}

pub struct Array([i32; 8]);
#[repr(packed)]
pub struct BigPacked1 {
    dealign: u8,
    data: Array
}

#[repr(packed(2))]
pub struct BigPacked2 {
    dealign: u8,
    data: Array
}

// CHECK-LABEL: @call_pkd1
#[no_mangle]
pub fn call_pkd1(f: fn() -> Array) -> BigPacked1 {
// CHECK: [[ALLOCA:%[_a-z0-9]+]] = alloca %Array
// CHECK: call void %{{.*}}(%Array* noalias nocapture noundef sret{{.*}} dereferenceable(32) [[ALLOCA]])
// CHECK: call void @llvm.memcpy.{{.*}}(i8* align 1 %{{.*}}, i8* align 4 %{{.*}}, i{{[0-9]+}} 32, i1 false)
    // check that calls whose destination is a field of a packed struct
    // go through an alloca rather than calling the function with an
    // unaligned destination.
    BigPacked1 { dealign: 0, data: f() }
}

// CHECK-LABEL: @call_pkd2
#[no_mangle]
pub fn call_pkd2(f: fn() -> Array) -> BigPacked2 {
// CHECK: [[ALLOCA:%[_a-z0-9]+]] = alloca %Array
// CHECK: call void %{{.*}}(%Array* noalias nocapture noundef sret{{.*}} dereferenceable(32) [[ALLOCA]])
// CHECK: call void @llvm.memcpy.{{.*}}(i8* align 2 %{{.*}}, i8* align 4 %{{.*}}, i{{[0-9]+}} 32, i1 false)
    // check that calls whose destination is a field of a packed struct
    // go through an alloca rather than calling the function with an
    // unaligned destination.
    BigPacked2 { dealign: 0, data: f() }
}

// CHECK-LABEL: @write_packed_array1
// CHECK: store i32 0, i32* %{{.+}}, align 1
// CHECK: store i32 1, i32* %{{.+}}, align 1
// CHECK: store i32 2, i32* %{{.+}}, align 1
#[no_mangle]
pub fn write_packed_array1(p: &mut BigPacked1) {
    p.data.0[0] = 0;
    p.data.0[1] = 1;
    p.data.0[2] = 2;
}

// CHECK-LABEL: @write_packed_array2
// CHECK: store i32 0, i32* %{{.+}}, align 2
// CHECK: store i32 1, i32* %{{.+}}, align 2
// CHECK: store i32 2, i32* %{{.+}}, align 2
#[no_mangle]
pub fn write_packed_array2(p: &mut BigPacked2) {
    p.data.0[0] = 0;
    p.data.0[1] = 1;
    p.data.0[2] = 2;
}

// CHECK-LABEL: @repeat_packed_array1
// CHECK: store i32 42, i32* %{{.+}}, align 1
#[no_mangle]
pub fn repeat_packed_array1(p: &mut BigPacked1) {
    p.data.0 = [42; 8];
}

// CHECK-LABEL: @repeat_packed_array2
// CHECK: store i32 42, i32* %{{.+}}, align 2
#[no_mangle]
pub fn repeat_packed_array2(p: &mut BigPacked2) {
    p.data.0 = [42; 8];
}

#[repr(packed)]
#[derive(Copy, Clone)]
pub struct Packed1Pair(u8, u32);

#[repr(packed(2))]
#[derive(Copy, Clone)]
pub struct Packed2Pair(u8, u32);

// CHECK-LABEL: @pkd1_pair
#[no_mangle]
pub fn pkd1_pair(pair1: &mut Packed1Pair, pair2: &mut Packed1Pair) {
// CHECK: call void @llvm.memcpy.{{.*}}(i8* align 1 %{{.*}}, i8* align 1 %{{.*}}, i{{[0-9]+}} 5, i1 false)
    *pair2 = *pair1;
}

// CHECK-LABEL: @pkd2_pair
#[no_mangle]
pub fn pkd2_pair(pair1: &mut Packed2Pair, pair2: &mut Packed2Pair) {
// CHECK: call void @llvm.memcpy.{{.*}}(i8* align 2 %{{.*}}, i8* align 2 %{{.*}}, i{{[0-9]+}} 6, i1 false)
    *pair2 = *pair1;
}

#[repr(packed)]
#[derive(Copy, Clone)]
pub struct Packed1NestedPair((u32, u32));

#[repr(packed(2))]
#[derive(Copy, Clone)]
pub struct Packed2NestedPair((u32, u32));

// CHECK-LABEL: @pkd1_nested_pair
#[no_mangle]
pub fn pkd1_nested_pair(pair1: &mut Packed1NestedPair, pair2: &mut Packed1NestedPair) {
// CHECK: call void @llvm.memcpy.{{.*}}(i8* align 1 %{{.*}}, i8* align 1 %{{.*}}, i{{[0-9]+}} 8, i1 false)
    *pair2 = *pair1;
}

// CHECK-LABEL: @pkd2_nested_pair
#[no_mangle]
pub fn pkd2_nested_pair(pair1: &mut Packed2NestedPair, pair2: &mut Packed2NestedPair) {
// CHECK: call void @llvm.memcpy.{{.*}}(i8* align 2 %{{.*}}, i8* align 2 %{{.*}}, i{{[0-9]+}} 8, i1 false)
    *pair2 = *pair1;
}
