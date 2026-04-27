//@ add-minicore
//@ assembly-output: emit-asm
//
//@ revisions: AVR
//@ [AVR] compile-flags: --target=avr-none -Ctarget-cpu=atmega328p
//@ [AVR] needs-llvm-components: avr
#![feature(c_variadic, no_core, lang_items, intrinsics, rustc_attrs)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[lang = "va_arg_safe"]
pub unsafe trait VaArgSafe {}

unsafe impl VaArgSafe for i16 {}
unsafe impl VaArgSafe for i32 {}
unsafe impl VaArgSafe for i64 {}
unsafe impl VaArgSafe for f32 {}
unsafe impl VaArgSafe for f64 {}
unsafe impl<T> VaArgSafe for *const T {}

#[repr(transparent)]
struct VaListInner {
    ptr: *const c_void,
}

#[repr(transparent)]
#[lang = "va_list"]
pub struct VaList<'a> {
    inner: VaListInner,
    _marker: PhantomData<&'a mut ()>,
}

#[rustc_intrinsic]
#[rustc_nounwind]
pub const unsafe fn va_arg<T: VaArgSafe>(ap: &mut VaList<'_>) -> T;

#[unsafe(no_mangle)]
unsafe extern "C" fn read_f32(ap: &mut VaList<'_>) -> f32 {
    // CHECK-LABEL: read_f32
    //
    // AVR: movw r30, r24
    // AVR-NEXT: ld r24, Z
    // AVR-NEXT: ldd r25, Z+1
    // AVR-NEXT: movw r26, r24
    // AVR-NEXT: adiw r26, 2
    // AVR-NEXT: std Z+1, r27
    // AVR-NEXT: st Z, r26
    // AVR-NEXT: movw r20, r30
    // AVR-NEXT: movw r18, r24
    // AVR-NEXT: movw r30, r18
    // AVR-NEXT: ld r22, Z
    // AVR-NEXT: ldd r23, Z+1
    // AVR-NEXT: adiw r24, 4
    // AVR-NEXT: movw r30, r20
    // AVR-NEXT: std Z+1, r25
    // AVR-NEXT: st Z, r24
    // AVR-NEXT: movw r30, r18
    // AVR-NEXT: ldd r24, Z+2
    // AVR-NEXT: ldd r25, Z+3
    // AVR-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_f64(ap: &mut VaList<'_>) -> f64 {
    // CHECK-LABEL: read_f64
    //
    // AVR: push r14
    // AVR-NEXT: push r15
    // AVR-NEXT: push r16
    // AVR-NEXT: push r17
    // AVR-NEXT: movw r30, r24
    // AVR-NEXT: ld r24, Z
    // AVR-NEXT: ldd r25, Z+1
    // AVR-NEXT: movw r26, r24
    // AVR-NEXT: adiw r26, 2
    // AVR-NEXT: std Z+1, r27
    // AVR-NEXT: st Z, r26
    // AVR-NEXT: movw r14, r30
    // AVR-NEXT: movw r16, r24
    // AVR-NEXT: movw r30, r16
    // AVR-NEXT: ld r18, Z
    // AVR-NEXT: ldd r19, Z+1
    // AVR-NEXT: movw r26, r24
    // AVR-NEXT: adiw r26, 4
    // AVR-NEXT: movw r30, r14
    // AVR-NEXT: std Z+1, r27
    // AVR-NEXT: st Z, r26
    // AVR-NEXT: movw r30, r16
    // AVR-NEXT: ldd r20, Z+2
    // AVR-NEXT: ldd r21, Z+3
    // AVR-NEXT: movw r26, r24
    // AVR-NEXT: adiw r26, 6
    // AVR-NEXT: movw r30, r14
    // AVR-NEXT: std Z+1, r27
    // AVR-NEXT: st Z, r26
    // AVR-NEXT: movw r30, r16
    // AVR-NEXT: ldd r22, Z+4
    // AVR-NEXT: ldd r23, Z+5
    // AVR-NEXT: adiw r24, 8
    // AVR-NEXT: movw r30, r14
    // AVR-NEXT: std Z+1, r25
    // AVR-NEXT: st Z, r24
    // AVR-NEXT: movw r30, r16
    // AVR-NEXT: ldd r24, Z+6
    // AVR-NEXT: ldd r25, Z+7
    // AVR-NEXT: pop r17
    // AVR-NEXT: pop r16
    // AVR-NEXT: pop r15
    // AVR-NEXT: pop r14
    // AVR-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i16(ap: &mut VaList<'_>) -> i16 {
    // CHECK-LABEL: read_i16
    //
    // AVR: movw r30, r24
    // AVR-NEXT: ld r24, Z
    // AVR-NEXT: ldd r25, Z+1
    // AVR-NEXT: movw r26, r24
    // AVR-NEXT: adiw r26, 2
    // AVR-NEXT: std Z+1, r27
    // AVR-NEXT: st Z, r26
    // AVR-NEXT: movw r30, r24
    // AVR-NEXT: ld r24, Z
    // AVR-NEXT: ldd r25, Z+1
    // AVR-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i32(ap: &mut VaList<'_>) -> i32 {
    // CHECK-LABEL: read_i32
    //
    // AVR: movw r30, r24
    // AVR-NEXT: ld r24, Z
    // AVR-NEXT: ldd r25, Z+1
    // AVR-NEXT: movw r26, r24
    // AVR-NEXT: adiw r26, 2
    // AVR-NEXT: std Z+1, r27
    // AVR-NEXT: st Z, r26
    // AVR-NEXT: movw r20, r30
    // AVR-NEXT: movw r18, r24
    // AVR-NEXT: movw r30, r18
    // AVR-NEXT: ld r22, Z
    // AVR-NEXT: ldd r23, Z+1
    // AVR-NEXT: adiw r24, 4
    // AVR-NEXT: movw r30, r20
    // AVR-NEXT: std Z+1, r25
    // AVR-NEXT: st Z, r24
    // AVR-NEXT: movw r30, r18
    // AVR-NEXT: ldd r24, Z+2
    // AVR-NEXT: ldd r25, Z+3
    // AVR-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i64(ap: &mut VaList<'_>) -> i64 {
    // CHECK-LABEL: read_i64
    //
    // AVR: push r14
    // AVR-NEXT: push r15
    // AVR-NEXT: push r16
    // AVR-NEXT: push r17
    // AVR-NEXT: movw r30, r24
    // AVR-NEXT: ld r24, Z
    // AVR-NEXT: ldd r25, Z+1
    // AVR-NEXT: movw r26, r24
    // AVR-NEXT: adiw r26, 2
    // AVR-NEXT: std Z+1, r27
    // AVR-NEXT: st Z, r26
    // AVR-NEXT: movw r14, r30
    // AVR-NEXT: movw r16, r24
    // AVR-NEXT: movw r30, r16
    // AVR-NEXT: ld r18, Z
    // AVR-NEXT: ldd r19, Z+1
    // AVR-NEXT: movw r26, r24
    // AVR-NEXT: adiw r26, 4
    // AVR-NEXT: movw r30, r14
    // AVR-NEXT: std Z+1, r27
    // AVR-NEXT: st Z, r26
    // AVR-NEXT: movw r30, r16
    // AVR-NEXT: ldd r20, Z+2
    // AVR-NEXT: ldd r21, Z+3
    // AVR-NEXT: movw r26, r24
    // AVR-NEXT: adiw r26, 6
    // AVR-NEXT: movw r30, r14
    // AVR-NEXT: std Z+1, r27
    // AVR-NEXT: st Z, r26
    // AVR-NEXT: movw r30, r16
    // AVR-NEXT: ldd r22, Z+4
    // AVR-NEXT: ldd r23, Z+5
    // AVR-NEXT: adiw r24, 8
    // AVR-NEXT: movw r30, r14
    // AVR-NEXT: std Z+1, r25
    // AVR-NEXT: st Z, r24
    // AVR-NEXT: movw r30, r16
    // AVR-NEXT: ldd r24, Z+6
    // AVR-NEXT: ldd r25, Z+7
    // AVR-NEXT: pop r17
    // AVR-NEXT: pop r16
    // AVR-NEXT: pop r15
    // AVR-NEXT: pop r14
    // AVR-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_ptr(ap: &mut VaList<'_>) -> *const u8 {
    // AVR: read_ptr = pm(read_i16)
    va_arg(ap)
}
