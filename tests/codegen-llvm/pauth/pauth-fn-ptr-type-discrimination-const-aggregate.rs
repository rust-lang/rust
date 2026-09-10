// ignore-tidy-file-linelength
//@ add-minicore
// Run it at O0, so that the compiler doesn't optimise the calls away.
//@ revisions: DISC NO_DISC
//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0
//
// Make sure that `OperandRef::from_const_alloc`'s `read_scalar` closure
// (compiler/rustc_codegen_ssa/src/mir/operand.rs), correctly signs function pointers with their
// discriminators.
// PAIR / PAIR_REV specifically exercise the `BackendRepr::ScalarPair` arm with the function
// pointer in each of the two field positions. And the offset used to key the discriminator
// lookup (0 for field `a`, `b_offset` for field `b`).
// TRIO forces `BackendRepr::Memory`.

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]
extern crate minicore;

extern "C" {
    fn f(ctx: *mut i32) -> i32;
}

type FnPtr = unsafe extern "C" fn(*mut i32) -> i32;

#[used]
// Baseline.
// DISC: @{{.*}}T_PTR = constant ptr ptrauth (ptr @{{.*}}f, i32 0, i64 12410), align 8
// NO_DISC: @{{.*}}T_PTR = constant ptr ptrauth (ptr @{{.*}}f, i32 0), align 8
static T_PTR: FnPtr = f;

// ScalarPair, function pointer at field `a` (local offset 0).
const PAIR: (FnPtr, i32) = (f, 1);

#[repr(C)]
struct PairRev {
    x: i32,
    f: FnPtr,
}

// ScalarPair, function pointer at field `b`
const PAIR_REV: PairRev = PairRev { x: 2, f };

// BackendRepr::Memory
// DISC: private unnamed_addr constant <{ ptr, [8 x i8] }> <{ ptr ptrauth (ptr @{{.*}}f, i32 0, i64 12410), [8 x i8]
// NO_DISC: private unnamed_addr constant <{ ptr, [8 x i8] }> <{ ptr ptrauth (ptr @{{.*}}f, i32 0), [8 x i8]
const TRIO: (FnPtr, i32, i32) = (f, 3, 4);

// CHECK-LABEL: main
pub fn main() {
    let mut x = 42i32;
    unsafe {
        // DISC: call i32 ptrauth (ptr @f, i32 0, i64 12410)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @f, i32 0)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_PTR(&mut x as *mut i32);

        // DISC: call i32 ptrauth (ptr @f, i32 0, i64 12410)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @f, i32 0)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = PAIR.0(&mut x as *mut i32);

        // DISC: call i32 ptrauth (ptr @f, i32 0, i64 12410)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @f, i32 0)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = (PAIR_REV.f)(&mut x as *mut i32);

        // DISC: call i32 %{{.*}}(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 %{{.*}}(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = TRIO.0(&mut x as *mut i32);
    }
}
