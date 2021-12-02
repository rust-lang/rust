// Test SPARC64 ABI
// - float structure members are passes in floating point registers
// (#86163)

// assembly-output: emit-asm
// needs-llvm-components: sparc
// compile-flags: --target=sparcv9-sun-solaris -Copt-level=3
#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "sized"]
pub trait Sized {}
#[lang = "copy"]
pub trait Copy {}

#[repr(C)]
pub struct Franta {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
}

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
}

extern "C" {
    fn opaque_callee(arg: Franta, intarg: i32);
    fn tst_use(arg: f32);
    fn clobber();
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
}
