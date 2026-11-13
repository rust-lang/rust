//@ add-minicore
//@ assembly-output: emit-asm
//
//@ compile-flags: -Copt-level=3 --target=avr-none -Ctarget-cpu=atmega328p
//@ needs-llvm-components: avr
#![feature(c_variadic_experimental_arch, no_core, lang_items, intrinsics, rustc_attrs)]
#![no_core]
#![crate_type = "lib"]

// Check that rustc and clang output match, see https://godbolt.org/z/1MvxoceeT.

extern crate minicore;
use minicore::ffi::VaList;
use minicore::*;

#[unsafe(no_mangle)]
unsafe extern "C" fn read_f32(ap: &mut VaList<'_>) -> f32 {
    // CHECK-LABEL: read_f32
    //
    // CHECK: movw r30, r24
    // CHECK-NEXT: ld r24, Z
    // CHECK-NEXT: ldd r25, Z+1
    // CHECK-NEXT: movw r26, r24
    // CHECK-NEXT: adiw r26, 4
    // CHECK-NEXT: std Z+1, r27
    // CHECK-NEXT: st Z, r26
    // CHECK-NEXT: movw r30, r24
    // CHECK-NEXT: ld r22, Z
    // CHECK-NEXT: ldd r23, Z+1
    // CHECK-NEXT: ldd r24, Z+2
    // CHECK-NEXT: ldd r25, Z+3
    // CHECK-NEXT: ret
    ap.next_arg()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_f64(ap: &mut VaList<'_>) -> f64 {
    // CHECK-LABEL: read_f64
    //
    // CHECK: movw r30, r24
    // CHECK-NEXT: ld r24, Z
    // CHECK-NEXT: ldd r25, Z+1
    // CHECK-NEXT: movw r26, r24
    // CHECK-NEXT: adiw r26, 8
    // CHECK-NEXT: std Z+1, r27
    // CHECK-NEXT: st Z, r26
    // CHECK-NEXT: movw r30, r24
    // CHECK-NEXT: ld r18, Z
    // CHECK-NEXT: ldd r19, Z+1
    // CHECK-NEXT: ldd r20, Z+2
    // CHECK-NEXT: ldd r21, Z+3
    // CHECK-NEXT: ldd r22, Z+4
    // CHECK-NEXT: ldd r23, Z+5
    // CHECK-NEXT: ldd r24, Z+6
    // CHECK-NEXT: ldd r25, Z+7
    // CHECK-NEXT: ret
    ap.next_arg()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i16(ap: &mut VaList<'_>) -> i16 {
    // CHECK-LABEL: read_i16
    //
    // CHECK: movw r30, r24
    // CHECK-NEXT: ld r24, Z
    // CHECK-NEXT: ldd r25, Z+1
    // CHECK-NEXT: movw r26, r24
    // CHECK-NEXT: adiw r26, 2
    // CHECK-NEXT: std Z+1, r27
    // CHECK-NEXT: st Z, r26
    // CHECK-NEXT: movw r30, r24
    // CHECK-NEXT: ld r24, Z
    // CHECK-NEXT: ldd r25, Z+1
    // CHECK-NEXT: ret
    ap.next_arg()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i32(ap: &mut VaList<'_>) -> i32 {
    // CHECK-LABEL: read_i32
    //
    // CHECK: movw r30, r24
    // CHECK-NEXT: ld r24, Z
    // CHECK-NEXT: ldd r25, Z+1
    // CHECK-NEXT: movw r26, r24
    // CHECK-NEXT: adiw r26, 4
    // CHECK-NEXT: std Z+1, r27
    // CHECK-NEXT: st Z, r26
    // CHECK-NEXT: movw r30, r24
    // CHECK-NEXT: ld r22, Z
    // CHECK-NEXT: ldd r23, Z+1
    // CHECK-NEXT: ldd r24, Z+2
    // CHECK-NEXT: ldd r25, Z+3
    // CHECK-NEXT: ret
    ap.next_arg()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i64(ap: &mut VaList<'_>) -> i64 {
    // CHECK-LABEL: read_i64
    //
    // CHECK: movw r30, r24
    // CHECK-NEXT: ld r24, Z
    // CHECK-NEXT: ldd r25, Z+1
    // CHECK-NEXT: movw r26, r24
    // CHECK-NEXT: adiw r26, 8
    // CHECK-NEXT: std Z+1, r27
    // CHECK-NEXT: st Z, r26
    // CHECK-NEXT: movw r30, r24
    // CHECK-NEXT: ld r18, Z
    // CHECK-NEXT: ldd r19, Z+1
    // CHECK-NEXT: ldd r20, Z+2
    // CHECK-NEXT: ldd r21, Z+3
    // CHECK-NEXT: ldd r22, Z+4
    // CHECK-NEXT: ldd r23, Z+5
    // CHECK-NEXT: ldd r24, Z+6
    // CHECK-NEXT: ldd r25, Z+7
    // CHECK-NEXT: ret
    ap.next_arg()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_ptr(ap: &mut VaList<'_>) -> *const u8 {
    // CHECK: read_ptr = pm(read_i16)
    ap.next_arg()
}
