// ignore-tidy-linelength
//@ revisions:aarch64 loongarch64 powerpc64 sparc64 x86_64
//@ min-llvm-version: 19
//@ compile-flags: -O -Cno-prepopulate-passes -Zlint-llvm-ir -Cllvm-args=-lint-abort-on-error

//@[aarch64] compile-flags: --target aarch64-unknown-linux-gnu
//@[aarch64] needs-llvm-components: arm
//@[loongarch64] compile-flags: --target loongarch64-unknown-linux-gnu
//@[loongarch64] needs-llvm-components: loongarch
//@[powerpc64] compile-flags: --target powerpc64-unknown-linux-gnu
//@[powerpc64] needs-llvm-components: powerpc
//@[sparc64] compile-flags: --target sparc64-unknown-linux-gnu
//@[sparc64] needs-llvm-components: sparc
//@[x86_64] compile-flags: --target x86_64-unknown-linux-gnu
//@[x86_64] needs-llvm-components: x86

// Tests that arguments with `PassMode::Cast` are handled correctly.

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_std]
#![no_core]

#[lang = "sized"]
trait Sized {}
#[lang = "freeze"]
trait Freeze {}
#[lang = "copy"]
trait Copy {}

// This struct will be passed as a single `i64` or `i32`.
// This may be (if `i64)) larger than the Rust layout, which is just `{ i16, i16 }`.
#[repr(C)]
pub struct TwoU16s {
    a: u16,
    b: u16,
}

// This struct will be passed as `[2 x i64]`.
// This is larger than the Rust layout.
#[repr(C)]
pub struct FiveU16s {
    a: u16,
    b: u16,
    c: u16,
    d: u16,
    e: u16,
}

// This struct will be passed as `[2 x double]`.
// This is the same as the Rust layout.
#[repr(C)]
pub struct DoubleDouble {
    f: f64,
    g: f64,
}

// On loongarch, this struct will be passed as `{ double, float }`.
// This is smaller than the Rust layout, which has trailing padding (`{ f64, f32, <f32 padding> }`)
#[repr(C)]
pub struct DoubleFloat {
    f: f64,
    g: f32,
}

// On x86_64, this struct will be passed as `{ i64, i32 }`.
// The load and store instructions will access 16 bytes, so we should allocate 16 bytes.
#[repr(C)]
pub struct Three32s {
    a: u32,
    b: u32,
    c: u32,
}

// CHECK-LABEL: @receives_twou16s
// aarch64-SAME:     ([[ABI_TYPE:i64]] {{.*}}[[ABI_VALUE:%.+]])
// loongarch64-SAME: ([[ABI_TYPE:i64]] {{.*}}[[ABI_VALUE:%.+]])
// powerpc64-SAME:   ([[ABI_TYPE:i32]] {{.*}}[[ABI_VALUE:%.+]])
// sparc64-SAME:     ([[ABI_TYPE:i64]] {{.*}}[[ABI_VALUE:%.+]])
// x86_64-SAME:      ([[ABI_TYPE:i32]] {{.*}}[[ABI_VALUE:%.+]])
#[no_mangle]
#[inline(never)]
pub extern "C" fn receives_twou16s(x: TwoU16s) {
    // CHECK-NEXT: [[START:.*:]]

    // aarch64-NEXT:     [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:8]]
    // loongarch64-NEXT: [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:8]]
    // sparc64-NEXT:     [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:8]]

    // CHECK-NEXT: [[RUST_ALLOCA:%.+]] = alloca [4 x i8], align [[RUST_ALIGN:2]]

    // aarch64-NEXT:     call void @llvm.lifetime.start.p0(i64 8, ptr [[ABI_ALLOCA]])
    // loongarch64-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[ABI_ALLOCA]])
    // sparc64-NEXT:     call void @llvm.lifetime.start.p0(i64 8, ptr [[ABI_ALLOCA]])
    //
    // aarch64-NEXT:     store i64 [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align 8
    // loongarch64-NEXT: store i64 [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align 8
    // sparc64-NEXT:     store i64 [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align 8
    //
    // aarch64-NEXT:     [[RUST_VALUE:%.+]] = load [[RUST_TYPE:%TwoU16s]], ptr [[ABI_ALLOCA]], align 8
    // loongarch64-NEXT: [[RUST_VALUE:%.+]] = load [[RUST_TYPE:%TwoU16s]], ptr [[ABI_ALLOCA]], align 8
    // sparc64-NEXT:     [[RUST_VALUE:%.+]] = load [[RUST_TYPE:%TwoU16s]], ptr [[ABI_ALLOCA]], align 8
    //
    // aarch64-NEXT:     call void @llvm.lifetime.end.p0(i64 8, ptr [[ABI_ALLOCA]])
    // loongarch64-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[ABI_ALLOCA]])
    // sparc64-NEXT:     call void @llvm.lifetime.end.p0(i64 8, ptr [[ABI_ALLOCA]])

    // aarch64-NEXT:     store [[RUST_TYPE]] [[RUST_VALUE]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // loongarch64-NEXT: store [[RUST_TYPE]] [[RUST_VALUE]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // powerpc64-NEXT:   store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // sparc64-NEXT:     store [[RUST_TYPE]] [[RUST_VALUE]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // x86_64-NEXT:      store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]

    // CHECK-NEXT: ret void
}

// CHECK-LABEL: @returns_twou16s
// powerpc64-SAME: sret([4 x i8]) align [[RUST_ALIGN:2]] {{.*}}[[RET_PTR:%.*]])
#[no_mangle]
#[inline(never)]
pub extern "C" fn returns_twou16s() -> TwoU16s {
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.
    // The other targets copy the cast ABI type to an alloca.

    // aarch64:     [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:2]]
    // loongarch64: [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:2]]
    // sparc64:     [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:2]]
    // x86_64:      [[ABI_ALLOCA:%.+]] = alloca [4 x i8], align [[ABI_ALIGN:2]]

    // aarch64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:i64]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // loongarch64: [[ABI_VALUE:%.+]] = load [[ABI_TYPE:i64]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // sparc64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:i64]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // x86_64:      [[ABI_VALUE:%.+]] = load [[ABI_TYPE:i32]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // aarch64:     ret [[ABI_TYPE]] [[ABI_VALUE]]
    // loongarch64: ret [[ABI_TYPE]] [[ABI_VALUE]]
    // sparc64:     ret [[ABI_TYPE]] [[ABI_VALUE]]
    // x86_64:      ret [[ABI_TYPE]] [[ABI_VALUE]]
    TwoU16s { a: 0, b: 1 }
}

// CHECK-LABEL: @receives_fiveu16s
// aarch64-SAME:     ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// loongarch64-SAME: ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// powerpc64-SAME:   ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// sparc64-SAME:     ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// x86_64-SAME:      ([[ABI_TYPE:{ i64, i16 }]] {{.*}}[[ABI_VALUE:%.+]])
#[no_mangle]
#[inline(never)]
pub extern "C" fn receives_fiveu16s(x: FiveU16s) {
    // CHECK-NEXT: [[START:.*:]]
    // CHECK-NEXT: [[RUST_ALLOCA:%.*]] = alloca [10 x i8], align [[RUST_ALIGN:2]]
    // CHECK-NEXT: [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    // CHECK-NEXT: store i64 [[TMP1]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN:2]]
    // CHECK-NEXT: [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    // CHECK-NEXT: [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // CHECK-NEXT: store {{i64|i16}} [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN:2]]
    // CHECK-NEXT: ret void
}

// CHECK-LABEL: @returns_fiveu16s
// powerpc64-SAME: sret([10 x i8]) align [[RUST_ALIGN:2]] {{.*}}[[RET_PTR:%.*]])
#[no_mangle]
#[inline(never)]
pub extern "C" fn returns_fiveu16s() -> FiveU16s {
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.
    // The other targets copy the cast ABI type to an alloca.

    // aarch64:     [[RUST_ALLOCA:%.+]] = alloca [10 x i8], align [[RUST_ALIGN:2]]
    // loongarch64: [[RUST_ALLOCA:%.+]] = alloca [10 x i8], align [[RUST_ALIGN:2]]
    // sparc64:     [[RUST_ALLOCA:%.+]] = alloca [10 x i8], align [[RUST_ALIGN:2]]
    // x86_64:      [[RUST_ALLOCA:%.+]] = alloca [10 x i8], align [[RUST_ALIGN:2]]

    // aarch64:     [[TMP4:%.*]] = load i64, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // loongarch64: [[TMP4:%.*]] = load i64, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // sparc64:     [[TMP4:%.*]] = load i64, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // x86_64:      [[TMP4:%.*]] = load i64, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    //
    // aarch64-NEXT:     [[TMP5:%.*]] = insertvalue [[ABI_TYPE:\[2 x i64\]]] poison, i64 [[TMP4]], 0
    // loongarch64-NEXT: [[TMP5:%.*]] = insertvalue [[ABI_TYPE:\[2 x i64\]]] poison, i64 [[TMP4]], 0
    // sparc64-NEXT:     [[TMP5:%.*]] = insertvalue [[ABI_TYPE:\[2 x i64\]]] poison, i64 [[TMP4]], 0
    // x86_64-NEXT:      [[TMP5:%.*]] = insertvalue [[ABI_TYPE:{ i64, i16 }]] poison, i64 [[TMP4]], 0
    //
    // aarch64-NEXT:     [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // loongarch64-NEXT: [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // sparc64-NEXT:     [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // x86_64-NEXT:      [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    //
    // aarch64-NEXT:     [[TMP7:%.*]] = load {{i64|i16}}, ptr [[TMP6]], align [[RUST_ALIGN]]
    // loongarch64-NEXT: [[TMP7:%.*]] = load {{i64|i16}}, ptr [[TMP6]], align [[RUST_ALIGN]]
    // sparc64-NEXT:     [[TMP7:%.*]] = load {{i64|i16}}, ptr [[TMP6]], align [[RUST_ALIGN]]
    // x86_64-NEXT:      [[TMP7:%.*]] = load {{i64|i16}}, ptr [[TMP6]], align [[RUST_ALIGN]]
    //
    // aarch64-NEXT:     [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], i64 [[TMP7]], 1
    // loongarch64-NEXT: [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], i64 [[TMP7]], 1
    // sparc64-NEXT:     [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], i64 [[TMP7]], 1
    // x86_64-NEXT:      [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], i16 [[TMP7]], 1
    //
    // aarch64-NEXT:     ret [[ABI_TYPE]] [[ABI_VALUE]]
    // loongarch64-NEXT: ret [[ABI_TYPE]] [[ABI_VALUE]]
    // sparc64-NEXT:     ret [[ABI_TYPE]] [[ABI_VALUE]]
    // x86_64-NEXT:      ret [[ABI_TYPE]] [[ABI_VALUE]]
    FiveU16s { a: 0, b: 1, c: 2, d: 3, e: 4 }
}

// CHECK-LABEL: @receives_doubledouble
// aarch64-SAME:     ([[ABI_TYPE:\[2 x double\]]] {{.*}}[[ABI_VALUE:%.+]])
// loongarch64-SAME: ([[ABI_TYPE:{ double, double }]] {{.*}}[[ABI_VALUE:%.+]])
// powerpc64-SAME:   ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// sparc64-SAME:     ([[ABI_TYPE:{ double, double }]] {{.*}}[[ABI_VALUE:%.+]])
// x86_64-SAME:      ([[ABI_TYPE:{ double, double }]] {{.*}}[[ABI_VALUE:%.+]])
#[no_mangle]
#[inline(never)]
pub extern "C" fn receives_doubledouble(x: DoubleDouble) {
    // CHECK-NEXT: [[START:.*:]]
    // CHECK-NEXT: [[RUST_ALLOCA:%.*]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // CHECK-NEXT: [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    // CHECK-NEXT: store {{double|i64}} [[TMP1]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // CHECK-NEXT: [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    // CHECK-NEXT: [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 [[RUST_ALIGN]]
    // CHECK-NEXT: store {{double|i64}} [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]
    // CHECK-NEXT: ret void
}

// CHECK-LABEL: @returns_doubledouble
// powerpc64-SAME: sret([16 x i8]) align [[RUST_ALIGN:8]] {{.*}}[[RET_PTR:%.*]])
#[no_mangle]
#[inline(never)]
pub extern "C" fn returns_doubledouble() -> DoubleDouble {
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.
    // The other targets copy the cast ABI type to an alloca.

    // aarch64:     [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // loongarch64: [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // sparc64:     [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // x86_64:      [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]

    // aarch64:     [[TMP4:%.*]] = load double, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // loongarch64: [[TMP4:%.*]] = load double, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // sparc64:     [[TMP4:%.*]] = load double, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // x86_64:      [[TMP4:%.*]] = load double, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    //
    // aarch64-NEXT:     [[TMP5:%.*]] = insertvalue [[ABI_TYPE:\[2 x double\]]] poison, double [[TMP4]], 0
    // loongarch64-NEXT: [[TMP5:%.*]] = insertvalue [[ABI_TYPE:{ double, double }]] poison, double [[TMP4]], 0
    // sparc64-NEXT:     [[TMP5:%.*]] = insertvalue [[ABI_TYPE:{ double, double }]] poison, double [[TMP4]], 0
    // x86_64-NEXT:      [[TMP5:%.*]] = insertvalue [[ABI_TYPE:{ double, double }]] poison, double [[TMP4]], 0
    //
    // aarch64-NEXT:     [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // loongarch64-NEXT: [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // sparc64-NEXT:     [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // x86_64-NEXT:      [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    //
    // aarch64-NEXT:     [[TMP7:%.*]] = load double, ptr [[TMP6]], align [[RUST_ALIGN]]
    // loongarch64-NEXT: [[TMP7:%.*]] = load double, ptr [[TMP6]], align [[RUST_ALIGN]]
    // sparc64-NEXT:     [[TMP7:%.*]] = load double, ptr [[TMP6]], align [[RUST_ALIGN]]
    // x86_64-NEXT:      [[TMP7:%.*]] = load double, ptr [[TMP6]], align [[RUST_ALIGN]]
    //
    // aarch64-NEXT:     [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], double [[TMP7]], 1
    // loongarch64-NEXT: [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], double [[TMP7]], 1
    // sparc64-NEXT:     [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], double [[TMP7]], 1
    // x86_64-NEXT:      [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], double [[TMP7]], 1
    //
    // aarch64-NEXT:     ret [[ABI_TYPE]] [[ABI_VALUE]]
    // loongarch64-NEXT: ret [[ABI_TYPE]] [[ABI_VALUE]]
    // sparc64-NEXT:     ret [[ABI_TYPE]] [[ABI_VALUE]]
    // x86_64-NEXT:      ret [[ABI_TYPE]] [[ABI_VALUE]]
    DoubleDouble { f: 0., g: 1. }
}

// CHECK-LABEL: @receives_three32s
// aarch64-SAME:     ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// loongarch64-SAME: ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// powerpc64-SAME:   ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// sparc64-SAME:     ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// x86_64-SAME:      ([[ABI_TYPE:{ i64, i32 }]] {{.*}}[[ABI_VALUE:%.+]])
#[no_mangle]
#[inline(never)]
pub extern "C" fn receives_three32s(x: Three32s) {
    // CHECK-NEXT: [[START:.*:]]
    // CHECK-NEXT: [[RUST_ALLOCA:%.*]] = alloca [12 x i8], align [[RUST_ALIGN:4]]
    // CHECK-NEXT: [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    // CHECK-NEXT: store i64 [[TMP1]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // CHECK-NEXT: [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    // CHECK-NEXT: [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // CHECK-NEXT: store {{i64|i32}} [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]
    // CHECK-NEXT: ret void
}

// CHECK-LABEL: @returns_three32s
// powerpc64-SAME: sret([12 x i8]) align [[RUST_ALIGN:4]] {{.*}}[[RET_PTR:%.*]])
#[no_mangle]
#[inline(never)]
pub extern "C" fn returns_three32s() -> Three32s {
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.
    // The other targets copy the cast ABI type to an alloca.

    // aarch64:     [[RUST_ALLOCA:%.+]] = alloca [12 x i8], align [[RUST_ALIGN:4]]
    // loongarch64: [[RUST_ALLOCA:%.+]] = alloca [12 x i8], align [[RUST_ALIGN:4]]
    // sparc64:     [[RUST_ALLOCA:%.+]] = alloca [12 x i8], align [[RUST_ALIGN:4]]
    // x86_64:      [[RUST_ALLOCA:%.+]] = alloca [12 x i8], align [[RUST_ALIGN:4]]

    // aarch64:     [[TMP4:%.*]] = load i64, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // loongarch64: [[TMP4:%.*]] = load i64, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // sparc64:     [[TMP4:%.*]] = load i64, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // x86_64:      [[TMP4:%.*]] = load i64, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    //
    // aarch64-NEXT:     [[TMP5:%.*]] = insertvalue [[ABI_TYPE:\[2 x i64\]]] poison, i64 [[TMP4]], 0
    // loongarch64-NEXT: [[TMP5:%.*]] = insertvalue [[ABI_TYPE:\[2 x i64\]]] poison, i64 [[TMP4]], 0
    // sparc64-NEXT:     [[TMP5:%.*]] = insertvalue [[ABI_TYPE:\[2 x i64\]]] poison, i64 [[TMP4]], 0
    // x86_64-NEXT:      [[TMP5:%.*]] = insertvalue [[ABI_TYPE:{ i64, i32 }]] poison, i64 [[TMP4]], 0
    //
    // aarch64-NEXT:     [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // loongarch64-NEXT: [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // sparc64-NEXT:     [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // x86_64-NEXT:      [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    //
    // aarch64-NEXT:     [[TMP7:%.*]] = load i64, ptr [[TMP6]], align [[RUST_ALIGN]]
    // loongarch64-NEXT: [[TMP7:%.*]] = load i64, ptr [[TMP6]], align [[RUST_ALIGN]]
    // sparc64-NEXT:     [[TMP7:%.*]] = load i64, ptr [[TMP6]], align [[RUST_ALIGN]]
    // x86_64-NEXT:      [[TMP7:%.*]] = load i32, ptr [[TMP6]], align [[RUST_ALIGN]]
    //
    // aarch64-NEXT:     [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], i64 [[TMP7]], 1
    // loongarch64-NEXT: [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], i64 [[TMP7]], 1
    // sparc64-NEXT:     [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], i64 [[TMP7]], 1
    // x86_64-NEXT:      [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], i32 [[TMP7]], 1
    //
    // aarch64-NEXT:     ret [[ABI_TYPE]] [[ABI_VALUE]]
    // loongarch64-NEXT: ret [[ABI_TYPE]] [[ABI_VALUE]]
    // sparc64-NEXT:     ret [[ABI_TYPE]] [[ABI_VALUE]]
    // x86_64-NEXT:      ret [[ABI_TYPE]] [[ABI_VALUE]]
    Three32s { a: 0, b: 0, c: 0 }
}

// These functions cause an ICE in sparc64 ABI code (https://github.com/rust-lang/rust/issues/122620)
#[cfg(not(target_arch = "sparc64"))]
// aarch64-LABEL:     @receives_doublefloat
// loongarch64-LABEL: @receives_doublefloat
// powerpc64-LABEL:   @receives_doublefloat
// x86_64-LABEL:      @receives_doublefloat

// aarch64-SAME:     ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// loongarch64-SAME: ([[ABI_TYPE:{ double, float }]] {{.*}}[[ABI_VALUE:%.+]])
// powerpc64-SAME:   ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// x86_64-SAME:      ([[ABI_TYPE:{ double, double }]] {{.*}}[[ABI_VALUE:%.+]])
#[no_mangle]
#[inline(never)]
pub extern "C" fn receives_doublefloat(x: DoubleFloat) {
    // aarch64-NEXT:     [[START:.*:]]
    // loongarch64-NEXT: [[START:.*:]]
    // powerpc64-NEXT:   [[START:.*:]]
    // x86_64-NEXT:      [[START:.*:]]

    // aarch64-NEXT:     [[RUST_ALLOCA:%.*]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // loongarch64-NEXT: [[RUST_ALLOCA:%.*]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // powerpc64-NEXT:   [[RUST_ALLOCA:%.*]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // x86_64-NEXT:      [[RUST_ALLOCA:%.*]] = alloca [16 x i8], align [[RUST_ALIGN:8]]

    // aarch64-NEXT:     [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    // loongarch64-NEXT: [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    // powerpc64-NEXT:   [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    // x86_64-NEXT:      [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0

    // aarch64-NEXT:     store i64 [[TMP1]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // loongarch64-NEXT: store double [[TMP1]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // powerpc64-NEXT:   store i64 [[TMP1]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // x86_64-NEXT:      store double [[TMP1]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]

    // aarch64-NEXT:     [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    // loongarch64-NEXT: [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    // powerpc64-NEXT:   [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    // x86_64-NEXT:      [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1

    // aarch64-NEXT:     [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // loongarch64-NEXT: [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // powerpc64-NEXT:   [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // x86_64-NEXT:      [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8

    // aarch64-NEXT:     store i64 [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]
    // loongarch64-NEXT: store float [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]
    // powerpc64-NEXT:   store i64 [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]
    // x86_64-NEXT:      store double [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]

    // aarch64-NEXT:     ret void
    // loongarch64-NEXT: ret void
    // powerpc64-NEXT:   ret void
    // x86_64-NEXT:      ret void
}

#[cfg(not(target_arch = "sparc64"))]
// aarch64-LABEL:     @returns_doublefloat
// loongarch64-LABEL: @returns_doublefloat
// powerpc64-LABEL:   @returns_doublefloat
// x86_64-LABEL:      @returns_doublefloat

// powerpc64-SAME: sret([16 x i8]) align [[RUST_ALIGN:8]] {{.*}}[[RET_PTR:%.*]])
#[no_mangle]
#[inline(never)]
pub extern "C" fn returns_doublefloat() -> DoubleFloat {
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.
    // The other targets copy the cast ABI type to an alloca.

    // aarch64:     [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // loongarch64: [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // x86_64:      [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]

    // aarch64:     [[TMP4:%.*]] = load i64, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // loongarch64: [[TMP4:%.*]] = load double, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // x86_64:      [[TMP4:%.*]] = load double, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    //
    // aarch64-NEXT:     [[TMP5:%.*]] = insertvalue [[ABI_TYPE:\[2 x i64\]]] poison, i64 [[TMP4]], 0
    // loongarch64-NEXT: [[TMP5:%.*]] = insertvalue [[ABI_TYPE:{ double, float }]] poison, double [[TMP4]], 0
    // x86_64-NEXT:      [[TMP5:%.*]] = insertvalue [[ABI_TYPE:{ double, double }]] poison, double [[TMP4]], 0

    // aarch64-NEXT:     [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // loongarch64-NEXT: [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // x86_64-NEXT:      [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    //
    // aarch64-NEXT:     [[TMP7:%.*]] = load i64, ptr [[TMP6]], align [[RUST_ALIGN]]
    // loongarch64-NEXT: [[TMP7:%.*]] = load float, ptr [[TMP6]], align [[RUST_ALIGN]]
    // x86_64-NEXT:      [[TMP7:%.*]] = load double, ptr [[TMP6]], align [[RUST_ALIGN]]
    //
    // aarch64-NEXT:     [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], i64 [[TMP7]], 1
    // loongarch64-NEXT: [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], float [[TMP7]], 1
    // x86_64-NEXT:      [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], double [[TMP7]], 1
    //
    // aarch64-NEXT:     ret [[ABI_TYPE]] [[ABI_VALUE]]
    // loongarch64-NEXT: ret [[ABI_TYPE]] [[ABI_VALUE]]
    // x86_64-NEXT:      ret [[ABI_TYPE]] [[ABI_VALUE]]
    DoubleFloat { f: 0., g: 0. }
}

// CHECK-LABEL: @call_twou16s
#[no_mangle]
pub fn call_twou16s() {
    // CHECK-NEXT: [[START:.*:]]
    // CHECK-NEXT: [[RUST_ALLOCA:%.*]] = alloca [4 x i8], align [[RUST_ALIGN:2]]
    // CHECK-NEXT: store i16 1, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // CHECK-NEXT: [[TMP1:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 2
    // CHECK-NEXT: store i16 2, ptr [[TMP1]], align [[RUST_ALIGN]]

    // aarch64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:i64]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // loongarch64: [[ABI_VALUE:%.+]] = load [[ABI_TYPE:i64]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // powerpc64:   [[ABI_VALUE:%.+]] = load [[ABI_TYPE:i32]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // sparc64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:i64]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // x86_64:      [[ABI_VALUE:%.+]] = load [[ABI_TYPE:i32]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]

    // CHECK-NEXT: call void @receives_twou16s([[ABI_TYPE]] [[ABI_VALUE]])
    // CHECK-NEXT: ret void
    let x = TwoU16s { a: 1, b: 2 };
    receives_twou16s(x);
}

// CHECK-LABEL: @return_twou16s
#[no_mangle]
pub fn return_twou16s() -> TwoU16s {
    // CHECK-NEXT: [[START:.*:]]
    //
    // aarch64-NEXT:     [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:8]]
    // loongarch64-NEXT: [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:8]]
    // sparc64-NEXT:     [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:8]]
    //
    // CHECK-NEXT: [[RUST_ALLOCA:%.+]] = alloca [4 x i8], align [[RUST_ALIGN:2]]
    //
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.

    // powerpc64: call void @returns_twou16s(ptr {{.+}} [[RUST_ALLOCA]])

    // The other targets copy the cast ABI type to by extractvalue.

    // aarch64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE:i64]] @returns_twou16s()
    // loongarch64: [[ABI_VALUE:%.+]] = call [[ABI_TYPE:i64]] @returns_twou16s()
    // sparc64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE:i64]] @returns_twou16s()
    // x86_64:      [[ABI_VALUE:%.+]] = call [[ABI_TYPE:i32]] @returns_twou16s()

    // aarch64:     call void @llvm.lifetime.start.p0(i64 8, ptr [[ABI_ALLOCA]])
    // loongarch64: call void @llvm.lifetime.start.p0(i64 8, ptr [[ABI_ALLOCA]])
    // sparc64:     call void @llvm.lifetime.start.p0(i64 8, ptr [[ABI_ALLOCA]])
    //
    // aarch64-NEXT:     store i64 [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align 8
    // loongarch64-NEXT: store i64 [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align 8
    // sparc64-NEXT:     store i64 [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align 8

    // aarch64-NEXT:     [[RUST_VALUE:%.+]] = load [[RUST_TYPE:%TwoU16s]], ptr [[ABI_ALLOCA]], align 8
    // loongarch64-NEXT: [[RUST_VALUE:%.+]] = load [[RUST_TYPE:%TwoU16s]], ptr [[ABI_ALLOCA]], align 8
    // sparc64-NEXT:     [[RUST_VALUE:%.+]] = load [[RUST_TYPE:%TwoU16s]], ptr [[ABI_ALLOCA]], align 8
    //
    // aarch64-NEXT:     call void @llvm.lifetime.end.p0(i64 8, ptr [[ABI_ALLOCA]])
    // loongarch64-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[ABI_ALLOCA]])
    // sparc64-NEXT:     call void @llvm.lifetime.end.p0(i64 8, ptr [[ABI_ALLOCA]])

    // aarch64-NEXT:     store [[RUST_TYPE]] [[RUST_VALUE]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // loongarch64-NEXT: store [[RUST_TYPE]] [[RUST_VALUE]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // sparc64-NEXT:     store [[RUST_TYPE]] [[RUST_VALUE]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // x86_64:           store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]

    returns_twou16s()
    // %_0.0 = load i16, ptr %0, align 2, !noundef !3
}

// CHECK-LABEL: @call_fiveu16s
#[no_mangle]
pub fn call_fiveu16s() {
    // CHECK: [[RUST_ALLOCA:%.+]] = alloca [10 x i8], align [[RUST_ALIGN:2]]
    // CHECK: [[TMP4:%.*]] = load i64, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]

    // aarch64-NEXT:     [[TMP5:%.*]] = insertvalue [[ABI_TYPE:\[2 x i64\]]] poison, i64 [[TMP4]], 0
    // loongarch64-NEXT: [[TMP5:%.*]] = insertvalue [[ABI_TYPE:\[2 x i64\]]] poison, i64 [[TMP4]], 0
    // powerpc64-NEXT:   [[TMP5:%.*]] = insertvalue [[ABI_TYPE:\[2 x i64\]]] poison, i64 [[TMP4]], 0
    // sparc64-NEXT:     [[TMP5:%.*]] = insertvalue [[ABI_TYPE:\[2 x i64\]]] poison, i64 [[TMP4]], 0
    // x86_64-NEXT:      [[TMP5:%.*]] = insertvalue [[ABI_TYPE:{ i64, i16 }]] poison, i64 [[TMP4]], 0

    // CHECK-NEXT: [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // CHECK-NEXT: [[TMP7:%.*]] = load {{i64|i16}}, ptr [[TMP6]], align [[RUST_ALIGN]]
    // CHECK-NEXT: [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], {{i64|i16}} [[TMP7]], 1
    // CHECK-NEXT: call void @receives_fiveu16s([[ABI_TYPE]] [[ABI_VALUE]])
    // CHECK-NEXT: ret void
    let x = FiveU16s { a: 1, b: 2, c: 3, d: 4, e: 5 };
    receives_fiveu16s(x);
}

// CHECK-LABEL: @return_fiveu16s
// CHECK-SAME: (ptr {{.+}} sret([10 x i8]) align [[RUST_ALIGN:2]] dereferenceable(10) [[RET_PTR:%.+]])
#[no_mangle]
pub fn return_fiveu16s() -> FiveU16s {
    // CHECK-NEXT: [[START:.*:]]
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.

    // powerpc64-NEXT: call void @returns_fiveu16s(ptr {{.+}} [[RET_PTR]])

    // The other targets copy the cast ABI type to the sret pointer.

    // aarch64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE:\[2 x i64\]]] @returns_fiveu16s()
    // loongarch64: [[ABI_VALUE:%.+]] = call [[ABI_TYPE:\[2 x i64\]]] @returns_fiveu16s()
    // sparc64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE:\[2 x i64\]]] @returns_fiveu16s()
    // x86_64:      [[ABI_VALUE:%.+]] = call [[ABI_TYPE:{ i64, i16 }]] @returns_fiveu16s()
    //
    // aarch64-NEXT:     [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    // loongarch64-NEXT: [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    // sparc64-NEXT:     [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    // x86_64-NEXT:      [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    //
    // aarch64-NEXT:     store i64 [[TMP1]], ptr [[RET_PTR]], align [[RUST_ALIGN]]
    // loongarch64-NEXT: store i64 [[TMP1]], ptr [[RET_PTR]], align [[RUST_ALIGN]]
    // sparc64-NEXT:     store i64 [[TMP1]], ptr [[RET_PTR]], align [[RUST_ALIGN]]
    // x86_64-NEXT:      store i64 [[TMP1]], ptr [[RET_PTR]], align [[RUST_ALIGN]]
    //
    // aarch64-NEXT:     [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    // loongarch64-NEXT: [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    // sparc64-NEXT:     [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    // x86_64-NEXT:      [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    //
    // aarch64-NEXT:     [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RET_PTR]], i64 8
    // loongarch64-NEXT: [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RET_PTR]], i64 8
    // sparc64-NEXT:     [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RET_PTR]], i64 8
    // x86_64-NEXT:      [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RET_PTR]], i64 8
    //
    // aarch64-NEXT:     store i64 [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]
    // loongarch64-NEXT: store i64 [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]
    // sparc64-NEXT:     store i64 [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]
    // x86_64-NEXT:      store i16 [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]

    // CHECK-NEXT: ret void
    returns_fiveu16s()
}

// CHECK-LABEL: @call_doubledouble
#[no_mangle]
pub fn call_doubledouble() {
    // CHECK: [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // CHECK: [[TMP4:%.*]] = load {{double|i64}}, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]

    // aarch64-NEXT:     [[TMP5:%.*]] = insertvalue [[ABI_TYPE:\[2 x double\]]] poison, double [[TMP4]], 0
    // loongarch64-NEXT: [[TMP5:%.*]] = insertvalue [[ABI_TYPE:{ double, double }]] poison, double [[TMP4]], 0
    // powerpc64-NEXT:   [[TMP5:%.*]] = insertvalue [[ABI_TYPE:\[2 x i64\]]] poison, i64 [[TMP4]], 0
    // sparc64-NEXT:     [[TMP5:%.*]] = insertvalue [[ABI_TYPE:{ double, double }]] poison, double [[TMP4]], 0
    // x86_64-NEXT:     [[TMP5:%.*]] = insertvalue [[ABI_TYPE:{ double, double }]] poison, double [[TMP4]], 0

    // CHECK-NEXT: [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // CHECK-NEXT: [[TMP7:%.*]] = load {{double|i64}}, ptr [[TMP6]], align [[RUST_ALIGN]]
    // CHECK-NEXT: [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], {{i64|double}} [[TMP7]], 1
    // CHECK-NEXT: call void @receives_doubledouble([[ABI_TYPE]] [[ABI_VALUE]])
    // CHECK-NEXT: ret void
    let x = DoubleDouble { f: 1., g: 2. };
    receives_doubledouble(x);
}

// CHECK-LABEL: @return_doubledouble
#[no_mangle]
pub fn return_doubledouble() -> DoubleDouble {
    // CHECK-NEXT: [[START:.*:]]
    // CHECK-NEXT: [[RET_PTR:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.

    // powerpc64: call void @returns_doubledouble(ptr {{.+}} [[RET_PTR]])

    // The other targets copy the cast ABI type to an alloca.

    // aarch64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE:\[2 x double\]]] @returns_doubledouble()
    // loongarch64: [[ABI_VALUE:%.+]] = call [[ABI_TYPE:{ double, double }]] @returns_doubledouble()
    // sparc64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE:{ double, double }]] @returns_doubledouble()
    // x86_64:      [[ABI_VALUE:%.+]] = call [[ABI_TYPE:{ double, double }]] @returns_doubledouble()

    // aarch64:     [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    // loongarch64: [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    // sparc64:     [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    // x86_64:      [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    //
    // aarch64-NEXT:     store double [[TMP1]], ptr [[RET_PTR]], align [[RUST_ALIGN]]
    // loongarch64-NEXT: store double [[TMP1]], ptr [[RET_PTR]], align [[RUST_ALIGN]]
    // sparc64-NEXT:     store double [[TMP1]], ptr [[RET_PTR]], align [[RUST_ALIGN]]
    // x86_64-NEXT:      store double [[TMP1]], ptr [[RET_PTR]], align [[RUST_ALIGN]]
    //
    // aarch64-NEXT:     [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    // loongarch64-NEXT: [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    // sparc64-NEXT:     [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    // x86_64-NEXT:      [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    //
    // aarch64-NEXT:     [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RET_PTR]], i64 8
    // loongarch64-NEXT: [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RET_PTR]], i64 8
    // sparc64-NEXT:     [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RET_PTR]], i64 8
    // x86_64-NEXT:      [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RET_PTR]], i64 8
    //
    // aarch64-NEXT:     store double [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]
    // loongarch64-NEXT: store double [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]
    // sparc64-NEXT:     store double [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]
    // x86_64-NEXT:      store double [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]
    returns_doubledouble()
}

// This test causes an ICE in sparc64 ABI code (https://github.com/rust-lang/rust/issues/122620)
#[cfg(not(target_arch = "sparc64"))]
// aarch64-LABEL:     @call_doublefloat
// loongarch64-LABEL: @call_doublefloat
// powerpc64-LABEL:   @call_doublefloat
// x86_64-LABEL:      @call_doublefloat
#[no_mangle]
pub fn call_doublefloat() {
    // aarch64:     [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // loongarch64: [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // powerpc64:   [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // x86_64:      [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]

    // aarch64:     [[TMP4:%.*]] = load {{double|i64}}, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // loongarch64: [[TMP4:%.*]] = load {{double|i64}}, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // powerpc64:   [[TMP4:%.*]] = load {{double|i64}}, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]
    // x86_64:      [[TMP4:%.*]] = load {{double|i64}}, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]

    // aarch64-NEXT:     [[TMP5:%.*]] = insertvalue [[ABI_TYPE:\[2 x i64\]]] poison, i64 [[TMP4]], 0
    // loongarch64-NEXT: [[TMP5:%.*]] = insertvalue [[ABI_TYPE:{ double, float }]] poison, double [[TMP4]], 0
    // powerpc64-NEXT:   [[TMP5:%.*]] = insertvalue [[ABI_TYPE:\[2 x i64\]]] poison, i64 [[TMP4]], 0
    // x86_64-NEXT:      [[TMP5:%.*]] = insertvalue [[ABI_TYPE:{ double, double }]] poison, double [[TMP4]], 0

    // aarch64-NEXT:     [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // loongarch64-NEXT: [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // powerpc64-NEXT:   [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // x86_64-NEXT:      [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8

    // aarch64-NEXT:     [[TMP7:%.*]] = load {{double|i64|float}}, ptr [[TMP6]], align [[RUST_ALIGN]]
    // loongarch64-NEXT: [[TMP7:%.*]] = load {{double|i64|float}}, ptr [[TMP6]], align [[RUST_ALIGN]]
    // powerpc64-NEXT:   [[TMP7:%.*]] = load {{double|i64|float}}, ptr [[TMP6]], align [[RUST_ALIGN]]
    // x86_64-NEXT:      [[TMP7:%.*]] = load {{double|i64|float}}, ptr [[TMP6]], align [[RUST_ALIGN]]

    // aarch64-NEXT:     [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], i64 [[TMP7]], 1
    // loongarch64-NEXT: [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], float [[TMP7]], 1
    // powerpc64-NEXT:   [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], i64 [[TMP7]], 1
    // x86_64-NEXT:      [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], double [[TMP7]], 1

    // aarch64-NEXT:     call void @receives_doublefloat([[ABI_TYPE]] {{(inreg )?}}[[ABI_VALUE]])
    // loongarch64-NEXT: call void @receives_doublefloat([[ABI_TYPE]] {{(inreg )?}}[[ABI_VALUE]])
    // powerpc64-NEXT:   call void @receives_doublefloat([[ABI_TYPE]] {{(inreg )?}}[[ABI_VALUE]])
    // x86_64-NEXT:      call void @receives_doublefloat([[ABI_TYPE]] {{(inreg )?}}[[ABI_VALUE]])
    let x = DoubleFloat { f: 1., g: 2. };
    receives_doublefloat(x);
}

// This test causes an ICE in sparc64 ABI code (https://github.com/rust-lang/rust/issues/122620)
#[cfg(not(target_arch = "sparc64"))]
// aarch64-LABEL:     @return_doublefloat
// loongarch64-LABEL: @return_doublefloat
// powerpc64-LABEL:   @return_doublefloat
// x86_64-LABEL:      @return_doublefloat
#[no_mangle]
pub fn return_doublefloat() -> DoubleFloat {
    // aarch64:     [[RET_PTR:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // loongarch64: [[RET_PTR:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // powerpc64:   [[RET_PTR:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // x86_64:      [[RET_PTR:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.

    // powerpc64: call void @returns_doublefloat(ptr {{.+}} [[RET_PTR]])

    // The other targets copy the cast ABI type to an alloca.

    // aarch64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE:\[2 x i64\]]] @returns_doublefloat()
    // loongarch64: [[ABI_VALUE:%.+]] = call [[ABI_TYPE:{ double, float }]] @returns_doublefloat()
    // x86_64:      [[ABI_VALUE:%.+]] = call [[ABI_TYPE:{ double, double }]] @returns_doublefloat()

    // aarch64:     [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    // loongarch64: [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    // x86_64:      [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    //
    // aarch64-NEXT:     store i64 [[TMP1]], ptr [[RET_PTR]], align [[RUST_ALIGN]]
    // loongarch64-NEXT: store double [[TMP1]], ptr [[RET_PTR]], align [[RUST_ALIGN]]
    // x86_64-NEXT:      store double [[TMP1]], ptr [[RET_PTR]], align [[RUST_ALIGN]]
    //
    // aarch64-NEXT:     [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    // loongarch64-NEXT: [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    // x86_64-NEXT:      [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    //
    // aarch64-NEXT:     [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RET_PTR]], i64 8
    // loongarch64-NEXT: [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RET_PTR]], i64 8
    // x86_64-NEXT:      [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RET_PTR]], i64 8
    //
    // aarch64-NEXT:     store i64 [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]
    // loongarch64-NEXT: store float [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]
    // x86_64-NEXT:      store double [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]
    returns_doublefloat()
}

// CHECK-LABEL: @call_three32s
#[no_mangle]
pub fn call_three32s() {
    // CHECK: [[RUST_ALLOCA:%.+]] = alloca [12 x i8], align [[RUST_ALIGN:4]]
    // CHECK: [[TMP4:%.*]] = load i64, ptr [[RUST_ALLOCA]], align [[RUST_ALIGN]]

    // aarch64-NEXT:     [[TMP5:%.*]] = insertvalue [[ABI_TYPE:\[2 x i64\]]] poison, i64 [[TMP4]], 0
    // loongarch64-NEXT: [[TMP5:%.*]] = insertvalue [[ABI_TYPE:\[2 x i64\]]] poison, i64 [[TMP4]], 0
    // powerpc64-NEXT:   [[TMP5:%.*]] = insertvalue [[ABI_TYPE:\[2 x i64\]]] poison, i64 [[TMP4]], 0
    // sparc64-NEXT:     [[TMP5:%.*]] = insertvalue [[ABI_TYPE:\[2 x i64\]]] poison, i64 [[TMP4]], 0
    // x86_64-NEXT:     [[TMP5:%.*]] = insertvalue [[ABI_TYPE:{ i64, i32 }]] poison, i64 [[TMP4]], 0

    // CHECK-NEXT: [[TMP6:%.*]] = getelementptr inbounds i8, ptr [[RUST_ALLOCA]], i64 8
    // CHECK-NEXT: [[TMP7:%.*]] = load {{i64|i32}}, ptr [[TMP6]], align [[RUST_ALIGN]]
    // CHECK-NEXT: [[ABI_VALUE:%.*]] = insertvalue [[ABI_TYPE]] [[TMP5]], {{i64|i32}} [[TMP7]], 1
    // CHECK-NEXT: call void @receives_three32s([[ABI_TYPE]] [[ABI_VALUE]])
    let x = Three32s { a: 1, b: 2, c: 3 };
    receives_three32s(x);
}

// Regression test for #75839
// CHECK-LABEL: @return_three32s(
// CHECK-SAME: sret([12 x i8]) align [[RUST_ALIGN:4]] {{.*}}[[RUST_RETVAL:%.*]])
#[no_mangle]
pub fn return_three32s() -> Three32s {
    // CHECK-NEXT: [[START:.*:]]
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.

    // powerpc64-NEXT: call void @returns_three32s(ptr {{.+}} [[RUST_RETVAL]])

    // aarch64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE:\[2 x i64\]]] @returns_three32s()
    // loongarch64: [[ABI_VALUE:%.+]] = call [[ABI_TYPE:\[2 x i64\]]] @returns_three32s()
    // sparc64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE:\[2 x i64\]]] @returns_three32s()
    // x86_64:      [[ABI_VALUE:%.+]] = call [[ABI_TYPE:{ i64, i32 }]] @returns_three32s()

    // aarch64:     [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    // loongarch64: [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    // sparc64:     [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    // x86_64:      [[TMP1:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 0
    //
    // aarch64-NEXT:     store i64 [[TMP1]], ptr [[RUST_RETVAL]], align [[RUST_ALIGN]]
    // loongarch64-NEXT: store i64 [[TMP1]], ptr [[RUST_RETVAL]], align [[RUST_ALIGN]]
    // sparc64-NEXT:     store i64 [[TMP1]], ptr [[RUST_RETVAL]], align [[RUST_ALIGN]]
    // x86_64-NEXT:      store i64 [[TMP1]], ptr [[RUST_RETVAL]], align [[RUST_ALIGN]]
    //
    // aarch64-NEXT:     [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    // loongarch64-NEXT: [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    // sparc64-NEXT:     [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    // x86_64-NEXT:      [[TMP2:%.*]] = extractvalue [[ABI_TYPE]] [[ABI_VALUE]], 1
    //
    // aarch64-NEXT:     [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RUST_RETVAL]], i64 8
    // loongarch64-NEXT: [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RUST_RETVAL]], i64 8
    // sparc64-NEXT:     [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RUST_RETVAL]], i64 8
    // x86_64-NEXT:      [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[RUST_RETVAL]], i64 8
    //
    // aarch64-NEXT:     store i64 [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]
    // loongarch64-NEXT: store i64 [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]
    // sparc64-NEXT:     store i64 [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]
    // x86_64-NEXT:      store i32 [[TMP2]], ptr [[TMP3]], align [[RUST_ALIGN]]
    //
    // CHECK-NEXT: ret void
    returns_three32s()
}
