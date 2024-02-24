//@ only-x86_64
//@ compile-flags: -O -C no-prepopulate-passes --crate-type=lib

// On LLVM 17 and earlier LLVM's own data layout specifies that i128 has 8 byte alignment,
// while rustc wants it to have 16 byte alignment. This test checks that we handle this
// correctly.

// CHECK: %ScalarPair = type { i32, [3 x i32], i128 }
// CHECK: %Struct = type { i32, i32, [2 x i32], i128 }

#![feature(core_intrinsics)]

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ScalarPair {
    a: i32,
    b: i128,
}

#[no_mangle]
pub fn load(x: &ScalarPair) -> ScalarPair {
    // CHECK-LABEL: @load(
    // CHECK-SAME: align 16 dereferenceable(32) %x
    // CHECK:      [[A:%.*]] = load i32, ptr %x, align 16
    // CHECK-NEXT: [[GEP:%.*]] = getelementptr inbounds i8, ptr %x, i64 16
    // CHECK-NEXT: [[B:%.*]] = load i128, ptr [[GEP]], align 16
    // CHECK-NEXT: [[IV1:%.*]] = insertvalue { i32, i128 } poison, i32 [[A]], 0
    // CHECK-NEXT: [[IV2:%.*]] = insertvalue { i32, i128 } [[IV1]], i128 [[B]], 1
    // CHECK-NEXT: ret { i32, i128 } [[IV2]]
    *x
}

#[no_mangle]
pub fn store(x: &mut ScalarPair) {
    // CHECK-LABEL: @store(
    // CHECK-SAME: align 16 dereferenceable(32) %x
    // CHECK:      store i32 1, ptr %x, align 16
    // CHECK-NEXT: [[GEP:%.*]] = getelementptr inbounds i8, ptr %x, i64 16
    // CHECK-NEXT: store i128 2, ptr [[GEP]], align 16
    *x = ScalarPair { a: 1, b: 2 };
}

#[no_mangle]
pub fn alloca() {
    // CHECK-LABEL: @alloca(
    // CHECK:      [[X:%.*]] = alloca %ScalarPair, align 16
    // CHECK:      store i32 1, ptr %x, align 16
    // CHECK-NEXT: [[GEP:%.*]] = getelementptr inbounds i8, ptr %x, i64 16
    // CHECK-NEXT: store i128 2, ptr [[GEP]], align 16
    let mut x = ScalarPair { a: 1, b: 2 };
    store(&mut x);
}

#[no_mangle]
pub fn load_volatile(x: &ScalarPair) -> ScalarPair {
    // CHECK-LABEL: @load_volatile(
    // CHECK-SAME: align 16 dereferenceable(32) %x
    // CHECK:      [[TMP:%.*]] = alloca %ScalarPair, align 16
    // CHECK:      [[LOAD:%.*]] = load volatile %ScalarPair, ptr %x, align 16
    // CHECK-NEXT: store %ScalarPair [[LOAD]], ptr [[TMP]], align 16
    // CHECK-NEXT: [[A:%.*]] = load i32, ptr [[TMP]], align 16
    // CHECK-NEXT: [[GEP:%.*]] = getelementptr inbounds i8, ptr [[TMP]], i64 16
    // CHECK-NEXT: [[B:%.*]] = load i128, ptr [[GEP]], align 16
    unsafe { std::intrinsics::volatile_load(x) }
}

#[no_mangle]
pub fn transmute(x: ScalarPair) -> (std::mem::MaybeUninit<i128>, i128) {
    // CHECK-LABEL: define { i128, i128 } @transmute(i32 noundef %x.0, i128 noundef %x.1)
    // CHECK:       [[TMP:%.*]] = alloca { i128, i128 }, align 16
    // CHECK-NEXT:  store i32 %x.0, ptr [[TMP]], align 16
    // CHECK-NEXT:  [[GEP:%.*]] = getelementptr inbounds i8, ptr [[TMP]], i64 16
    // CHECK-NEXT:  store i128 %x.1, ptr [[GEP]], align 16
    // CHECK-NEXT:  [[LOAD1:%.*]] = load i128, ptr %_0, align 16
    // CHECK-NEXT:  [[GEP2:%.*]] = getelementptr inbounds i8, ptr [[TMP]], i64 16
    // CHECK-NEXT:  [[LOAD2:%.*]] = load i128, ptr [[GEP2]], align 16
    // CHECK-NEXT:  [[IV1:%.*]] = insertvalue { i128, i128 } poison, i128 [[LOAD1]], 0
    // CHECK-NEXT:  [[IV2:%.*]] = insertvalue { i128, i128 } [[IV1]], i128 [[LOAD2]], 1
    // CHECK-NEXT:  ret { i128, i128 } [[IV2]]
    unsafe { std::mem::transmute(x) }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Struct {
    a: i32,
    b: i32,
    c: i128,
}

#[no_mangle]
pub fn store_struct(x: &mut Struct) {
    // CHECK-LABEL: @store_struct(
    // CHECK-SAME: align 16 dereferenceable(32) %x
    // CHECK:      [[TMP:%.*]] = alloca %Struct, align 16
    // CHECK:      store i32 1, ptr [[TMP]], align 16
    // CHECK-NEXT: [[GEP1:%.*]] = getelementptr inbounds %Struct, ptr [[TMP]], i32 0, i32 1
    // CHECK-NEXT: store i32 2, ptr [[GEP1]], align 4
    // CHECK-NEXT: [[GEP2:%.*]] = getelementptr inbounds %Struct, ptr [[TMP]], i32 0, i32 3
    // CHECK-NEXT: store i128 3, ptr [[GEP2]], align 16
    // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %x, ptr align 16 [[TMP]], i64 32, i1 false)
    *x = Struct { a: 1, b: 2, c: 3 };
}
