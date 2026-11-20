// ignore-tidy-file-linelength
//@ add-minicore
// Run it at O0, so that the compiler doesn't optimise the calls away.

//@ revisions: DISC NO_DISC
//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0

// Check the signing of internal members of structs that are themselves function pointers.
// Transmuting never resigns: the bits are copied unchanged, so a call through the new type
// authenticates with the new type's discriminator (and traps at runtime if the signatures are not
// ABI-compatible, which is UB in Rust).

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
// DISC-DAG: @{{.*}}T_TREE_SRC = constant <{ ptr, ptr, ptr, ptr }> <{ ptr ptrauth (ptr @{{.*}}f, i32 0, i64 18983), ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712), ptr ptrauth (ptr @{{.*}}h, i32 0, i64 55265), ptr ptrauth (ptr @{{.*}}i, i32 0, i64 44485) }>
// NO_DISC-DAG: @{{.*}}T_TREE_SRC = constant <{ ptr, ptr, ptr, ptr }> <{ ptr ptrauth (ptr @{{.*}}f, i32 0), ptr ptrauth (ptr @{{.*}}g, i32 0), ptr ptrauth (ptr @{{.*}}h, i32 0), ptr ptrauth (ptr @{{.*}}i, i32 0) }>
// DISC-DAG: @{{.*}}T_WRAPPED_FN_PTR = internal constant ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712)
// NO_DISC-DAG: @{{.*}}T_WRAPPED_FN_PTR = internal constant ptr ptrauth (ptr @{{.*}}g, i32 0)

// DISC-DAG: @{{.*}}T_TUPLE = constant <{ ptr, ptr }> <{ ptr ptrauth (ptr @{{.*}}f, i32 0, i64 18983), ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712) }>
// NO_DISC-DAG: @{{.*}}T_TUPLE = constant <{ ptr, ptr }> <{ ptr ptrauth (ptr @{{.*}}f, i32 0), ptr ptrauth (ptr @{{.*}}g, i32 0) }>

// DISC-DAG: @{{.*}}T_TUPLE_2 = constant <{ ptr, ptr }> <{ ptr ptrauth (ptr @{{.*}}m, i32 0, i64 21613), ptr ptrauth (ptr @{{.*}}n, i32 0, i64 7718) }>
// NO_DISC-DAG: @{{.*}}T_TUPLE_2 = constant <{ ptr, ptr }> <{ ptr ptrauth (ptr @{{.*}}m, i32 0), ptr ptrauth (ptr @{{.*}}n, i32 0) }>

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

// Aggregate reinterpretation (the whole Struct, not just a Member) with mixed members.
#[inline(never)]
// CHECK-DAG: test_6_mixed_layout_cast
pub fn test_6_mixed_layout_cast() {
    let x = (A(f), NotFn(999));
    // DISC: store ptr ptrauth (ptr @{{.*}}f, i32 0, i64 18983), ptr
    // NO_DISC: [[Y:%.*]] = alloca [16 x i8]
    // NO_DISC: store ptr ptrauth (ptr @{{.*}}f, i32 0), ptr [[Y]]
    let y: (B, AlsoNotFn) = unsafe { mem::transmute(x) };

    unsafe {
        ptr::read_volatile(&y);
        // DISC-NOT: call i64 @llvm.ptrauth.resign
        // DISC: call void %{{.*}}(i32 42) {{.*}} [ "ptrauth"(i32 0, i64 2712) ]
        // NO_DISC: [[Y_LOAD:%.*]] = load ptr, ptr [[Y]]
        // NO_DISC: call void [[Y_LOAD]](i32 42) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        (y.0.0)(42);
    }
}

#[repr(transparent)]
struct Wrapper(extern "C" fn(i32));

impl Sync for Wrapper {}

static T_WRAPPED_FN_PTR: Wrapper = Wrapper(g);

#[repr(C)]
struct RootSrc {
    f0: extern "C" fn(),
    f1: extern "C" fn(i32),
    f2: extern "C" fn(i64, i64),
    f3: extern "C" fn(i64, i64, f32),
}

impl Sync for RootSrc {}

#[used]
static T_TREE_SRC: RootSrc = RootSrc { f0: f, f1: g, f2: h, f3: i };

type T0 = extern "C" fn();
type T1 = extern "C" fn(i32);

type T2 = extern "C" fn(f32);
type T3 = extern "C" fn(f64);

#[repr(C)]
struct TupleSrc {
    pair: (T0, T1),
}

#[repr(C)]
struct TupleDst {
    pair: (T2, T3),
}

impl Sync for TupleSrc {}
impl Sync for TupleDst {}

extern "C" fn m(f32: f32) {}
extern "C" fn n(f64: f64) {}
#[used]
static T_TUPLE: TupleSrc = TupleSrc { pair: (f, g) };
#[used]
static T_TUPLE_2: TupleDst = TupleDst { pair: (m, n) };

#[inline(never)]
// Tuple test
// CHECK-DAG: test_9_tuple
pub fn test_9_tuple() {
    let src = TupleSrc { pair: (f, g) };
    let _ = unsafe { ptr::read_volatile(&src) };
    // DISC: store ptr ptrauth (ptr @{{.*}}f, i32 0, i64 18983)
    // DISC: store ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712)
    // NO_DISC: store ptr ptrauth (ptr @{{.*}}f, i32 0)
    // NO_DISC: store ptr ptrauth (ptr @{{.*}}g, i32 0)
    // CHECK-NOT: call i64 @llvm.ptrauth.resign
    let dst: TupleDst = unsafe { mem::transmute(src) };

    // DISC: call void %{{.*}}(float {{.*}}) #[[#]] [ "ptrauth"(i32 0, i64 21613) ]
    // DISC: call void %{{.*}}(double {{.*}}) #[[#]] [ "ptrauth"(i32 0, i64 7718) ]
    // NO_DISC: call void %{{.*}}(float {{.*}}) #[[#]] [ "ptrauth"(i32 0, i64 0) ]
    // NO_DISC: call void %{{.*}}(double {{.*}}) #[[#]] [ "ptrauth"(i32 0, i64 0) ]
    (dst.pair.0)(0.42f32);
    (dst.pair.1)(0.4242f64);
}
