// Test SPARC64 ABI
// - float structure members are passes in floating point registers
// (#86163)

//@ add-minicore
//@ assembly-output: emit-asm
//@ needs-llvm-components: sparc
//@ compile-flags: --target=sparcv9-sun-solaris -Copt-level=3
#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]
#![feature(f128)]

extern crate minicore;
use minicore::*;

#[repr(C)]
pub struct Franta {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
}

#[repr(C, packed)]
struct Misaligned(i32, f64);

#[repr(C)]
struct AlignToMakeAssemblyShorter<T>(T, f64);

#[repr(C)]
pub struct Floats(i32, f32, f64, f128);

#[repr(C)]
pub struct LessFloats(f32, i32, f64);

#[repr(C)]
pub struct NotMisaligned(i32, Misaligned);

#[repr(C, align(16))]
pub struct Align16(f64, i32, i32);

impl Copy for Misaligned {}
impl<T: Copy> Copy for AlignToMakeAssemblyShorter<T> {}
impl Copy for Floats {}
impl Copy for LessFloats {}
impl Copy for NotMisaligned {}
impl Copy for Align16 {}

// NB: due to delay slots the `ld` following the call is actually executed before the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn callee(arg: Franta) {
    // CHECK-LABEL: callee:
    // CHECK: st %f3, [[PLACE_D:.*]]
    // CHECK: st %f2, [[PLACE_C:.*]]
    // CHECK: st %f1, [[PLACE_B:.*]]
    // CHECK: st %f0, [[PLACE_A:.*]]
    // CHECK: call tst_use
    // CHECK-NEXT: ld [[PLACE_A]], %f1
    // CHECK: call tst_use
    // CHECK-NEXT: ld [[PLACE_B]], %f1
    // CHECK: call tst_use
    // CHECK-NEXT: ld [[PLACE_C]], %f1
    // CHECK: call tst_use
    // CHECK-NEXT: ld [[PLACE_D]], %f1
    clobber();
    tst_use(arg.a);
    tst_use(arg.b);
    tst_use(arg.c);
    tst_use(arg.d);
    tail_call_avoidance_fn();
}

extern "C" {
    fn opaque_callee(arg: Franta, intarg: i32);
    fn tst_use(arg: f32);
    fn clobber();
    // This exists so that post-https://reviews.llvm.org/D138741 LLVM doesn't
    // tail-call away some of our assertions.
    fn tail_call_avoidance_fn();
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn caller() {
    // CHECK-LABEL: caller:
    // CHECK: ld [{{.*}}], %f0
    // CHECK: ld [{{.*}}], %f1
    // CHECK: ld [{{.*}}], %f2
    // CHECK: ld [{{.*}}], %f3
    // CHECK: call opaque_callee
    // CHECK: mov 3, %o2
    opaque_callee(Franta { a: 1.0, b: 2.0, c: 3.0, d: 4.0 }, 3);
    tail_call_avoidance_fn();
}

// Check that misaligned floats aren't promoted to floating point registers.
// CHECK-LABEL: misaligned_arg:
#[unsafe(no_mangle)]
extern "C" fn misaligned_arg(x: &mut AlignToMakeAssemblyShorter<Misaligned>, value: Misaligned) {
    // CHECK: srlx %o2, 32, %o2
    // CHECK-NEXT: stx %o1, [%o0]
    // CHECK-NEXT: retl
    // CHECK-NEXT: st %o2, [%o0+8]
    x.0 = value;
}

// CHECK-LABEL: misaligned_ret:
#[unsafe(no_mangle)]
extern "C" fn misaligned_ret(x: &AlignToMakeAssemblyShorter<Misaligned>) -> Misaligned {
    // CHECK: ld [%o0+8], %o1
    // CHECK-NEXT: ldx [%o0], %o0
    // CHECK-NEXT: retl
    // CHECK-NEXT: sllx %o1, 32, %o1
    x.0
}

// Check structs where 32 >= size > 16 are promoted to register only as an argument.
// Also check that the various floating-point types are promoted to the correct registers.
// CHECK-LABEL: floats_arg:
#[unsafe(no_mangle)]
extern "C" fn floats_arg(x: &mut Floats, value: Floats) {
    // CHECK: ldx [%o1+24], %o2
    // CHECK-NEXT: ldx [%o1+16], %o3
    // CHECK-NEXT: ldx [%o1+8], %o4
    // CHECK-NEXT: ldx [%o1], %o1
    // CHECK-NEXT: stx %o2, [%o0+24]
    // CHECK-NEXT: stx %o3, [%o0+16]
    // CHECK-NEXT: stx %o4, [%o0+8]
    // CHECK-NEXT: retl
    // CHECK-NEXT: stx %o1, [%o0]
    *x = value;
}

// CHECK-LABEL: floats_ret:
#[unsafe(no_mangle)]
extern "C" fn floats_ret(x: &Floats) -> Floats {
    // CHECK: ld [%o0+4], %f1
    // CHECK-NEXT: ldd [%o0+8], %f2
    // CHECK-NEXT: ldd [%o0+16], %f4
    // CHECK-NEXT: ld [%o0], %o1
    // CHECK-NEXT: ldd [%o0+24], %f6
    // CHECK-NEXT: retl
    // CHECK-NEXT: sllx %o1, 32, %o0
    *x
}

// Check float promotion when passing as an argument with a struct where size <= 16.
// CHECK-LABEL: less_floats_arg:
#[unsafe(no_mangle)]
extern "C" fn less_floats_arg(x: &mut LessFloats, value: LessFloats) {
    // CHECK: st %f2, [%o0]
    // CHECK-NEXT: st %o1, [%o0+4]
    // CHECK-NEXT: retl
    // CHECK-NEXT: std %f4, [%o0+8]
    *x = value;
}

// CHECK-LABEL: less_floats_ret:
#[unsafe(no_mangle)]
extern "C" fn less_floats_ret(x: &LessFloats) -> LessFloats {
    // CHECK: ld [%o0], %f0
    // CHECK-NEXT: ldd [%o0+8], %f2
    // CHECK-NEXT: retl
    // CHECK-NEXT: ld [%o0+4], %o0
    *x
}

// Check fields are promoted if they are aligned in the overall structure.
// This matches Clang's behaviour but not GCC's.
// CHECK-LABEL: not_misaligned_arg:
#[unsafe(no_mangle)]
extern "C" fn not_misaligned_arg(
    x: &mut AlignToMakeAssemblyShorter<NotMisaligned>,
    value: NotMisaligned,
) {
    // CHECK: stx %o1, [%o0]
    // CHECK-NEXT: retl
    // CHECK-NEXT: std %f4, [%o0+8]
    x.0 = value;
}

// CHECK-LABEL: not_misaligned_ret:
#[unsafe(no_mangle)]
extern "C" fn not_misaligned_ret(x: &AlignToMakeAssemblyShorter<NotMisaligned>) -> NotMisaligned {
    // CHECK: ldx [%o0], %o1
    // CHECK-NEXT: ldd [%o0+8], %f2
    // CHECK-NEXT: retl
    // CHECK-NEXT: mov %o1, %o0
    x.0
}

// Check that 16-aligned structs are allocated the correct registers.
// CHECK-LABEL: align_16_arg:
#[unsafe(no_mangle)]
extern "C" fn align_16_arg(x: &mut Align16, value: Align16) {
    // CHECK: std %f4, [%o0]
    // CHECK-NEXT: retl
    // CHECK-NEXT: stx %o3, [%o0+8]
    *x = value;
}

// CHECK-LABEL: align_16_ret:
#[unsafe(no_mangle)]
extern "C" fn align_16_ret(x: &Align16) -> Align16 {
    // CHECK: ldd [%o0], %f0
    // CHECK-NEXT: retl
    // CHECK-NEXT: ldx [%o0+8], %o1
    *x
}

// Check ZST args don't prevent further arguments from being processed.
// CHECK-LABEL: zst_arg:
#[unsafe(no_mangle)]
extern "C" fn zst_arg(_: (), value: LessFloats, x: &mut LessFloats) {
    // CHECK: st %f0, [%o2]
    // CHECK-NEXT: st %o0, [%o2+4]
    // CHECK-NEXT: retl
    // CHECK-NEXT: std %f2, [%o2+8]
    *x = value;
}

#[repr(C)]
struct I32F32Input {
    a: i32,
    b: f32,
}

#[repr(C)]
struct I32F32Output {
    b: f32,
    a: i32,
}

// The clang/LLVM implementation mentions that this case requires special handling.
// CHECK-LABEL: i32_f32:
#[unsafe(no_mangle)]
extern "C" fn i32_f32(input: I32F32Input) -> I32F32Output {
    // CHECK: srlx %o0, 32, %o0
    // CHECK-NEXT: fmovs %f1, %f0
    // CHECK-NEXT: retl
    // CHECK-NEXT: nop
    I32F32Output { a: input.a, b: input.b }
}

#[repr(C)]
pub struct C {
    a: f64,
    b: f32,
}

// regression test for https://github.com/rust-lang/rust/issues/147883.
#[unsafe(no_mangle)]
pub extern "C" fn foo(c: C) -> C {
    c
}
