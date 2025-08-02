// Test SPARC64 ABI
// - float structure members are passes in floating point registers
// (#86163)

//@ add-core-stubs
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
pub struct Floats(i32, f32, f64, f128);

#[repr(C)]
pub struct LessFloats(f32, i32, f64);

#[repr(C)]
pub struct NotMisaligned(i32, Misaligned);

#[repr(C, align(16))]
pub struct Align16(f64, i32, i32);

// NB: due to delay slots the `ld` following the call is actually executed before the call.
#[no_mangle]
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

#[no_mangle]
pub unsafe extern "C" fn caller() {
    // CHECK-LABEL: caller:
    // CHECK: ld [{{.*}}], %f0
    // CHECK: ld [{{.*}}], %f1
    // CHECK: ld [{{.*}}], %f2
    // CHECK: ld [{{.*}}], %f3
    // CHECK: call opaque_callee
    // CHECK: mov     3, %o2
    opaque_callee(Franta { a: 1.0, b: 2.0, c: 3.0, d: 4.0 }, 3);
    tail_call_avoidance_fn();
}

// CHECK-LABEL: misaligned:
#[no_mangle]
extern "C" fn misaligned(_: i64, value: Misaligned) -> Misaligned {
    // CHECK: sethi 4194303, %o0
    // CHECK-NEXT: xor %o0, -1024, %o0
    // CHECK-NEXT: and %o2, %o0, %o2
    // CHECK-NEXT: mov %o1, %o0
    // CHECK-NEXT: retl
    // CHECK-NEXT: mov %o2, %o1
    value
}

// CHECK-LABEL: floats:
#[no_mangle]
extern "C" fn floats(_: i64, value: Floats) -> Floats {
    // CHECK: ld [%o1+4], %f1
    // CHECK-NEXT: ldd [%o1+8], %f2
    // CHECK-NEXT: ldd [%o1+16], %f4
    // CHECK-NEXT: ld [%o1], %o0
    // CHECK-NEXT: ldd [%o1+24], %f6
    // CHECK-NEXT: retl
    // CHECK-NEXT: sllx %o0, 32, %o0
    value
}

// CHECK-LABEL: less_floats:
#[no_mangle]
extern "C" fn less_floats(_: i64, value: LessFloats) -> LessFloats {
    // CHECK: mov %o1, %o0
    // CHECK-NEXT: fmovs %f2, %f0
    // CHECK-NEXT: fmovd %f4, %f2
    // CHECK-NEXT: retl
    // CHECK-NEXT: nop
    value
}

// CHECK-LABEL: not_misaligned:
#[no_mangle]
extern "C" fn not_misaligned(_: i64, value: NotMisaligned) -> NotMisaligned {
    // CHECK: fmovd %f4, %f2
    // CHECK-NEXT: retl
    // CHECK-NEXT: mov %o1, %o0
    value
}

// CHECK-LABEL: align_16:
#[no_mangle]
extern "C" fn align_16(_: i64, value: Align16) -> Align16 {
    // CHECK: mov %o3, %o1
    // CHECK-NEXT: fmovd %f4, %f0
    // CHECK-NEXT: retl
    // CHECK-NEXT: nop
    value
}
