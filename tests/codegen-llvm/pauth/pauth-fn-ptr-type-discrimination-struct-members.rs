// ignore-tidy-file-linelength
//@ add-minicore
//@ only-pauthtest
// Run it at O0, so that the compiler doesn't optimise the calls away.

//@ revisions: DISC NO_DISC
//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0

// Test generation of function-pointer type discriminators. The discriminator values were obtained
// from Clang by compiling equivalent C code (included at the end of the file). Both compilers must
// generate identical values.
//
// Check the signing of internal members of structs that are themselves function pointers.

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::{Sync, mem, ptr};

// Function definitions, used as members in structs.
extern "C" fn f() {}
extern "C" fn g(i32: i32) {}
extern "C" fn h(i64: i64, j: i64) {}
extern "C" fn i(i64: i64, b: i64, c: f32) {}

// Structs...
#[repr(transparent)]
struct A(extern "C" fn());

#[repr(transparent)]
struct B(extern "C" fn(i32));

#[repr(transparent)]
struct C(extern "C" fn(i64, i64));

#[repr(transparent)]
struct NotFn(u64);

#[repr(transparent)]
struct AlsoNotFn(u64);

// and their wrappers (L - level).
#[repr(transparent)]
struct L1A(A);

#[repr(transparent)]
struct L1B(B);

#[repr(transparent)]
struct L2A(L1A);

#[repr(transparent)]
struct L2B(L1B);

#[repr(transparent)]
struct L3A(L2A);

#[repr(transparent)]
struct L3B(L2B);

#[repr(transparent)]
struct L4A(L3A);

#[repr(transparent)]
struct L4B(L3B);

#[repr(transparent)]
struct L5A(L4A);

#[repr(transparent)]
struct L5B(L4B);

#[repr(C)]
struct MixedPair {
    f0: extern "C" fn(),
    f1: extern "C" fn(i32),
}

#[repr(transparent)]
struct L1NotFn(NotFn);

#[repr(transparent)]
struct L1AlsoNotFn(AlsoNotFn);

// Make sure that static initialization traverses struct members and uses correct discriminators.
// DISC-DAG: @{{.*}}T_TREE_SRC = internal constant <{ ptr, ptr, ptr, ptr }> <{ ptr ptrauth (ptr @{{.*}}f, i32 0, i64 18983), ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712), ptr ptrauth (ptr @{{.*}}h, i32 0, i64 55265), ptr ptrauth (ptr @{{.*}}i, i32 0, i64 44485) }>
// NO_DISC-DAG: @{{.*}}T_TREE_SRC = internal constant <{ ptr, ptr, ptr, ptr }> <{ ptr ptrauth (ptr @{{.*}}f, i32 0), ptr ptrauth (ptr @{{.*}}g, i32 0), ptr ptrauth (ptr @{{.*}}h, i32 0), ptr ptrauth (ptr @{{.*}}i, i32 0) }>
// DISC-DAG: @{{.*}}T_WRAPPED_FN_PTR = internal constant ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712)
// NO_DISC-DAG: @{{.*}}T_WRAPPED_FN_PTR = internal constant ptr ptrauth (ptr @{{.*}}g, i32 0)

// Simplest fn ptr resign through a struct transmute.
#[inline(never)]
// CHECK-DAG: test_1_struct_resign
pub fn test_1_struct_resign() {
    let a: A = A(f);
    // DISC: [[PTR_RESIGNED:%.*]] = call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @{{.*}}f, i32 0, i64 18983) to i64), i32 0, i64 18983, i32 0, i64 2712)
    // DISC: [[INT_TO_PTR:%.*]] = inttoptr i64 [[PTR_RESIGNED]] to ptr
    // DISC: store ptr [[INT_TO_PTR]], ptr [[PTR_STORED:%*.]]
    // NO_DISC-NOT: call i64 @llvm.ptrauth.resign
    // NO_DISC: store ptr ptrauth (ptr @{{.*}}f, i32 0), ptr [[STORE:%.*]], align 8
    let b: B = unsafe { mem::transmute(a) };

    unsafe {
        // DISC: [[PTR_RELOADED:%.*]] = load ptr, ptr [[PTR_STORED]]
        // NO_DISC: [[LOAD:%.*]] = load ptr, ptr [[STORE]]
        ptr::read_volatile(&b);
        // DISC: call void [[PTR_RELOADED]](i32 42) {{.*}} [ "ptrauth"(i32 0, i64 2712) ]
        // NO_DISC: call void [[LOAD]](i32 42) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        (b.0)(42);
    }
}

// Same as above but in a deep chain, expect the chain to disappear.
#[inline(never)]
// CHECK-DAG: test_2_deep_nested
pub fn test_2_deep_nested() {
    let a = L5A(L4A(L3A(L2A(L1A(A(f))))));
    // DISC: [[PTR_RESIGNED:%.*]] = call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @{{.*}}f, i32 0, i64 18983) to i64), i32 0, i64 18983, i32 0, i64 2712)
    // DISC: [[INT_TO_PTR:%.*]] = inttoptr i64 [[PTR_RESIGNED]] to ptr
    // DISC: store ptr [[INT_TO_PTR]], ptr [[PTR_STORED:%*.]]
    // NO_DISC-NOT: call i64 @llvm.ptrauth.resign
    // NO_DISC: store ptr ptrauth (ptr @{{.*}}f, i32 0), ptr [[STORE:%.*]], align 8
    let b: L5B = unsafe { mem::transmute(a) };

    unsafe {
        // DISC: [[PTR_RELOADED:%.*]] = load ptr, ptr [[PTR_STORED]]
        // NO_DISC: [[LOAD:%.*]] = load ptr, ptr [[STORE]]
        ptr::read_volatile(&b);
        // DISC: call void [[PTR_RELOADED]](i32 4242) {{.*}} [ "ptrauth"(i32 0, i64 2712) ]
        // NO_DISC: call void [[LOAD]](i32 4242) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        (b.0.0.0.0.0.0)(4242);
    }
}

// Different destination discriminator.
#[inline(never)]
// CHECK-DAG: test_3_cross_fnptr_cast
pub fn test_3_cross_fnptr_cast() {
    let a: A = A(f);
    // DISC: [[PTR_RESIGNED:%.*]] = call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @{{.*}}f, i32 0, i64 18983) to i64), i32 0, i64 18983, i32 0, i64 55265)
    // DISC: [[INT_TO_PTR:%.*]] = inttoptr i64 [[PTR_RESIGNED]] to ptr
    // DISC: store ptr [[INT_TO_PTR]], ptr [[PTR_STORED:%*.]]
    // NO_DISC-NOT: call i64 @llvm.ptrauth.resign
    // NO_DISC: store ptr ptrauth (ptr @{{.*}}f, i32 0), ptr [[STORE:%.*]], align 8
    let c: C = unsafe { mem::transmute(a) };

    unsafe {
        // DISC: [[PTR_RELOADED:%.*]] = load ptr, ptr [[PTR_STORED]]
        // NO_DISC: [[LOAD:%.*]] = load ptr, ptr [[STORE]]
        ptr::read_volatile(&c);
        // DISC: call void [[PTR_RELOADED]](i64 1, i64 2) {{.*}} [ "ptrauth"(i32 0, i64 55265) ]
        // NO_DISC: call void [[LOAD]](i64 1, i64 2) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        (c.0)(1, 2);
    }
}

// Negative control.
#[inline(never)]
// CHECK-DAG: test_4_non_fnptr_cast
pub fn test_4_non_fnptr_cast() {
    // CHECK-NOT: llvm.ptrauth.resign
    // CHECK-NOT: ptrauth
    let x = L1NotFn(NotFn(123));
    let y: L1AlsoNotFn = unsafe { mem::transmute(x) };

    unsafe {
        ptr::read_volatile(&y);
    }
}

// Mixed resigned and non-resigned fields.
#[inline(never)]
// CHECK-DAG: test_5_mixed_fnptr_cast_resign
pub fn test_5_mixed_fnptr_cast_resign() {
    // Allocate the MixedPair.
    // DISC: [[M:%.*]] = alloca [16 x i8]
    let mut m = MixedPair {
        // Resign fn() -> fn(i32).
        // DISC: [[RESIGNED:%.*]] = call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @{{.*}}f, i32 0, i64 18983) to i64), i32 0, i64 18983, i32 0, i64 2712)
        // DISC: [[F1:%.*]] = inttoptr i64 [[RESIGNED]] to ptr
        // Store first struct member.
        // DISC: store ptr ptrauth (ptr @{{.*}}f, i32 0, i64 18983), ptr [[M]]
        // Compute address of second member and store it
        // DISC: [[M1:%.*]] = getelementptr inbounds i8, ptr [[M]], i64 8
        // DISC: store ptr [[F1]], ptr [[M1]], align 8
        // NO_DISC-NOT: call i64 @llvm.ptrauth.resign
        // NO_DISC: store ptr ptrauth (ptr @{{.*}}f, i32 0), ptr [[M:%.*]], align 8
        // NO_DISC: [[M_0:%.*]] = getelementptr inbounds i8, ptr [[M]], i64 8
        // NO_DISC: store ptr ptrauth (ptr @{{.*}}f, i32 0), ptr [[M_0]], align 8
        f0: f,
        f1: unsafe { mem::transmute::<extern "C" fn(), extern "C" fn(i32)>(f) },
    };

    // Volatile read of the whole struct.
    // DISC: [[PAIR:%.*]] = call { ptr, ptr } @{{.*}}read_volatile
    // NO_DISC: [[PAIR:%.*]] = call { ptr, ptr } @{{.*}}read_volatile
    let tmp = unsafe { ptr::read_volatile(&m) };

    // Extract both fields and call each of them
    // DISC: [[TMP0:%.*]] = extractvalue { ptr, ptr } [[PAIR]], 0
    // DISC: [[TMP1:%.*]] = extractvalue { ptr, ptr } [[PAIR]], 1
    // DISC: call void [[TMP0]]() {{.*}} "ptrauth"(i32 0, i64 18983)
    // DISC: call void [[TMP1]](i32 123) {{.*}} "ptrauth"(i32 0, i64 2712)
    // NO_DISC: [[TMP0:%.*]] = extractvalue { ptr, ptr } [[PAIR]], 0
    // NO_DISC: [[TMP1:%.*]] = extractvalue { ptr, ptr } [[PAIR]], 1
    // NO_DISC: call void {{.*}}() {{.*}} [ "ptrauth"(i32 0, i64 0) ]
    // NO_DISC: call void {{.*}}(i32 123) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
    (tmp.f0)();
    (tmp.f1)(123);
}

// Aggregate reinterpretation (the whole Struct, not just a Member) with mixed members.
#[inline(never)]
// CHECK-DAG: test_6_mixed_layout_cast
pub fn test_6_mixed_layout_cast() {
    let x = (A(f), NotFn(999));
    // DISC: [[Y:%.*]] = alloca [16 x i8]
    // DISC: store ptr ptrauth (ptr @{{.*}}f, i32 0, i64 18983), ptr [[Y]]
    // NO_DISC: [[Y:%.*]] = alloca [16 x i8]
    // NO_DISC: store ptr ptrauth (ptr @{{.*}}f, i32 0), ptr [[Y]]
    let y: (B, AlsoNotFn) = unsafe { mem::transmute(x) };

    unsafe {
        ptr::read_volatile(&y);
        // DISC: [[Y_LOAD:%.*]] = load ptr, ptr [[Y]]
        // DISC: call void [[Y_LOAD]](i32 42) {{.*}} [ "ptrauth"(i32 0, i64 2712) ]
        // NO_DISC: [[Y_LOAD:%.*]] = load ptr, ptr [[Y]]
        // NO_DISC: call void [[Y_LOAD]](i32 42) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        (y.0.0)(42);
    }
}

impl Sync for RootSrc {}
impl Sync for RootDst {}

#[repr(C)]
struct RootSrc {
    f0: extern "C" fn(),
    f1: extern "C" fn(i32),
    f2: extern "C" fn(i64, i64),
    f3: extern "C" fn(i64, i64, f32),
}

type G0 = extern "C" fn(i32);
type G1 = extern "C" fn(i64, i64);
type G2 = extern "C" fn(i64, i64, f32);
type G3 = extern "C" fn();

#[repr(C)]
struct RootDst {
    f0: G0,
    f1: G1,
    f2: G2,
    f3: G3,
}

static T_TREE_SRC: RootSrc = RootSrc { f0: f, f1: g, f2: h, f3: i };

#[inline(never)]
// Aggregate stress test, multiple resigning.
// CHECK-DAG: test_7_tree_cast_mixed
pub fn test_7_tree_cast_mixed() {
    let src: RootSrc = unsafe { ptr::read_volatile(&T_TREE_SRC) };
    let dst = RootDst {
        // field 0: load -> resign(18983 -> 2712) -> store
        // DISC: [[SRC0:%.*]] = load ptr, ptr [[SRC:%.*]],
        // DISC: [[SRC0I:%.*]] = ptrtoint ptr [[SRC0]] to i64
        // DISC: [[RESIGN0:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[SRC0I]], i32 0, i64 18983, i32 0, i64 2712)
        // DISC: [[DST0:%.*]] = inttoptr i64 [[RESIGN0]] to ptr
        f0: unsafe { mem::transmute::<extern "C" fn(), G0>(src.f0) },
        // field 1: load -> resign(2712 -> 55265) -> store
        // DISC: [[SRC1PTR:%.*]] = getelementptr inbounds i8, ptr [[SRC]], i64 8
        // DISC: [[SRC1:%.*]] = load ptr, ptr [[SRC1PTR]]
        // DISC: [[SRC1I:%.*]] = ptrtoint ptr [[SRC1]] to i64
        // DISC: [[RESIGN1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[SRC1I]], i32 0, i64 2712, i32 0, i64 55265)
        // DISC: [[DST1:%.*]] = inttoptr i64 [[RESIGN1]] to ptr
        f1: unsafe { mem::transmute::<extern "C" fn(i32), G1>(src.f1) },
        // field 2: load -> resign(5526 -> 44485) -> store
        // DISC: [[SRC2PTR:%.*]] = getelementptr inbounds i8, ptr [[SRC]], i64 16
        // DISC: [[SRC2:%.*]] = load ptr, ptr [[SRC2PTR]]
        // DISC: [[SRC2I:%.*]] = ptrtoint ptr [[SRC2]] to i64
        // DISC: [[RESIGN2:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[SRC2I]], i32 0, i64 55265, i32 0, i64 44485)
        // DISC: [[DST2:%.*]] = inttoptr i64 [[RESIGN2]] to ptr
        f2: unsafe { mem::transmute::<extern "C" fn(i64, i64), G2>(src.f2) },
        // field 3: load -> resign(44485 -> 18983) -> store
        // DISC: [[SRC3PTR:%.*]] = getelementptr inbounds i8, ptr [[SRC]], i64 24
        // DISC: [[SRC3:%.*]] = load ptr, ptr [[SRC3PTR]]
        // DISC: [[SRC3I:%.*]] = ptrtoint ptr [[SRC3]] to i64
        // DISC: [[RESIGN3:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[SRC3I]], i32 0, i64 44485, i32 0, i64 18983)
        // DISC: [[DST3:%.*]] = inttoptr i64 [[RESIGN3]] to ptr
        f3: unsafe { mem::transmute::<extern "C" fn(i64, i64, f32), G3>(src.f3) },
        // DISC: store ptr [[DST0]], ptr [[DST:%.*]],
        // DISC: [[DST1PTR:%.*]] = getelementptr inbounds i8, ptr %dst, i64 8
        // DISC: store ptr [[DST1]], ptr [[DST1PTR]]
        // DISC: [[DST2PTR:%.*]] = getelementptr inbounds i8, ptr %dst, i64 16
        // DISC: store ptr [[DST2]], ptr [[DST2PTR]]
        // DISC: [[DST3PTR:%.*]] = getelementptr inbounds i8, ptr %dst, i64 24
        // DISC: store ptr [[DST3]], ptr [[DST3PTR]]
    };

    unsafe {
        ptr::read_volatile(&dst);
    }
    // NO_DISC-NOT: call i64 @llvm.ptrauth.resign

    // Field loads and authed calls:
    // DISC: [[CALL0:%.*]] = load ptr, ptr [[DST]]
    // DISC: call void [[CALL0]](i32 1) {{.*}} [ "ptrauth"(i32 0, i64 2712) ]
    // NO_DISC: call void {{.*}}(i32 1) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
    (dst.f0)(1);
    // DISC: [[CALL1PTR:%.*]] = getelementptr inbounds i8, ptr [[DST]], i64 8
    // DISC: [[CALL1:%.*]] = load ptr, ptr [[CALL1PTR]],
    // DISC: call void [[CALL1]](i64 2, i64 3) {{.*}} [ "ptrauth"(i32 0, i64 55265) ]
    // NO_DISC: call void {{.*}}(i64 2, i64 3) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
    (dst.f1)(2, 3);
    // DISC: [[CALL2PTR:%.*]] = getelementptr inbounds i8, ptr [[DST]], i64 16
    // DISC: [[CALL2:%.*]] = load ptr, ptr [[CALL2PTR]],
    // DISC: call void [[CALL2]](i64 4, i64 5, float 6.000000e+00) {{.*}} [ "ptrauth"(i32 0, i64 44485) ]
    // NO_DISC: call void {{.*}}(i64 4, i64 5, float 6.000000e+00) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
    (dst.f2)(4, 5, 6.0);
    // DISC: [[CALL3PTR:%.*]] = getelementptr inbounds i8, ptr [[DST]], i64 24
    // DISC: [[CALL3:%.*]] = load ptr, ptr [[CALL3PTR]],
    // DISC: call void [[CALL3]]() {{.*}} [ "ptrauth"(i32 0, i64 18983) ]
    // NO_DISC: call void {{.*}}() {{.*}} [ "ptrauth"(i32 0, i64 0) ]
    (dst.f3)();
}

#[repr(transparent)]
struct Wrapper(extern "C" fn(i32));

impl Sync for Wrapper {}

static T_WRAPPED_FN_PTR: Wrapper = Wrapper(g);

// C equivalent.
// #include <stdint.h>
//
// typedef void (*fn0)(void);
// typedef void (*fn1)(int);
// typedef void (*fn2)(long long, long long);
// typedef void (*fn3)(long long, long long, float);
//
// typedef void (*g0)(int);
// typedef void (*g1)(long long, long long);
// typedef void (*g2)(long long, long long, float);
// typedef void (*g3)(void);
//
// void f(void);
// void g(int);
// void h(long long, long long);
// void i(long long, long long, float);
//
// typedef struct {
//   fn0 f;
// } A;
// typedef struct {
//   fn1 f;
// } B;
// typedef struct {
//   fn2 f;
// } C;
// typedef struct {
//   A a;
// } L1A;
// typedef struct {
//   B b;
// } L1B;
// typedef struct {
//   L1A a;
// } L2A;
// typedef struct {
//   L1B b;
// } L2B;
// typedef struct {
//   L2A a;
// } L3A;
// typedef struct {
//   L2B b;
// } L3B;
// typedef struct {
//   L3A a;
// } L4A;
// typedef struct {
//   L3B b;
// } L4B;
// typedef struct {
//   L4A a;
// } L5A;
// typedef struct {
//   L4B b;
// } L5B;
// typedef struct {
//   fn0 f0;
//   fn1 f1;
// } MixedPair;
// typedef struct {
//   uint64_t x;
// } NotFn;
// typedef struct {
//   uint64_t x;
// } AlsoNotFn;
// typedef struct {
//   NotFn n;
// } L1NotFn;
// typedef struct {
//   AlsoNotFn a;
// } L1AlsoNotFn;
//
// __attribute__((noinline)) void test_1_struct_resign(void) {
//   A a;
//   a.f = f;
//
//   B b;
//   b.f = (fn1)a.f;
//
//   volatile B tmp = b;
//   b.f(42);
// }
//
// __attribute__((noinline)) void test_2_deep_nested(void) {
//   L5A a;
//   a.a.a.a.a.a.f = f;
//
//   L5B b;
//   b.b.b.b.b.b.f = (fn1)a.a.a.a.a.a.f;
//
//   volatile L5B tmp = b;
//   b.b.b.b.b.b.f(42);
// }
//
// __attribute__((noinline)) void test_3_cross_fnptr_cast(void) {
//   A a;
//   a.f = f;
//
//   C c;
//   c.f = (fn2)a.f;
//
//   volatile C tmp = c;
//   c.f(1, 2);
// }
//
// __attribute__((noinline)) void test_4_non_fnptr_cast(void) {
//   L1NotFn x;
//   x.n.x = 123;
//
//   L1AlsoNotFn y;
//   y.a = *(AlsoNotFn *)&x;
//
//   volatile L1AlsoNotFn tmp = y;
// }
//
// __attribute__((noinline)) void test_5_mixed_fnptr_cast_resign(void) {
//   MixedPair m;
//   m.f0 = f;
//   m.f1 = (fn1)f;
//
//   volatile MixedPair tmp = m;
//   m.f0();
//   m.f1(123);
// }
//
// typedef struct {
//   A a;
//   NotFn n;
// } TupleA;
//
// typedef struct {
//   B b;
//   AlsoNotFn n;
// } TupleB;
//
// __attribute__((noinline)) void test_6_mixed_layout_cast(void) {
//   TupleA x;
//   x.a.f = f;
//   x.n.x = 999;
//
//   TupleB y = *(TupleB *)&x;
//
//   volatile TupleB tmp = y;
//
//   y.b.f(42);
// }
//
// typedef struct {
//   fn0 f0;
//   fn1 f1;
//   fn2 f2;
//   fn3 f3;
// } RootSrc;
//
// typedef struct {
//   g0 f0;
//   g1 f1;
//   g2 f2;
//   g3 f3;
// } RootDst;
//
// static const RootSrc T_TREE_SRC = {
//     .f0 = f,
//     .f1 = g,
//     .f2 = h,
//     .f3 = i,
// };
//
// __attribute__((noinline)) void test_7_tree_cast_mixed(void) {
//   volatile const RootSrc *vp = &T_TREE_SRC;
//
//   RootSrc src = *vp;
//
//   RootDst dst = {
//       .f0 = (g0)src.f0,
//       .f1 = (g1)src.f1,
//       .f2 = (g2)src.f2,
//       .f3 = (g3)src.f3,
//   };
//
//   volatile RootDst tmp = dst;
//
//   dst.f0(1);
//   dst.f1(2, 3);
//   dst.f2(4, 5, 6.0f);
//   dst.f3();
// }
