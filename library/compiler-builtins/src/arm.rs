use core::intrinsics;

#[cfg(feature = "mem")]
use mem::{memcpy, memmove, memset};

// NOTE This function and the ones below are implemented using assembly because they using a custom
// calling convention which can't be implemented using a normal Rust function
#[naked]
#[cfg_attr(not(test), no_mangle)]
#[cfg(not(feature = "gen-tests"))]
pub unsafe fn __aeabi_uidivmod() {
    asm!("push {lr}
          sub sp, sp, #4
          mov r2, sp
          bl __udivmodsi4
          ldr r1, [sp]
          add sp, sp, #4
          pop {pc}");
    intrinsics::unreachable();
}

#[naked]
#[cfg_attr(not(test), no_mangle)]
#[cfg(not(feature = "gen-tests"))]
pub unsafe fn __aeabi_uldivmod() {
    asm!("push {r4, lr}
          sub sp, sp, #16
          add r4, sp, #8
          str r4, [sp]
          bl __udivmoddi4
          ldr r2, [sp, #8]
          ldr r3, [sp, #12]
          add sp, sp, #16
          pop {r4, pc}");
    intrinsics::unreachable();
}

#[naked]
#[cfg_attr(not(test), no_mangle)]
#[cfg(not(feature = "gen-tests"))]
pub unsafe fn __aeabi_idivmod() {
    asm!("push {r0, r1, r4, lr}
          bl __divsi3
          pop {r1, r2}
          muls r2, r2, r0
          subs r1, r1, r2
          pop {r4, pc}");
    intrinsics::unreachable();
}

#[naked]
#[cfg_attr(not(test), no_mangle)]
#[cfg(not(feature = "gen-tests"))]
pub unsafe fn __aeabi_ldivmod() {
    asm!("push {r4, lr}
          sub sp, sp, #16
          add r4, sp, #8
          str r4, [sp]
          bl __divmoddi4
          ldr r2, [sp, #8]
          ldr r3, [sp, #12]
          add sp, sp, #16
          pop {r4, pc}");
    intrinsics::unreachable();
}

// TODO: These aeabi_* functions should be defined as aliases
#[cfg(not(feature = "mem"))]
extern "C" {
    fn memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8;
    fn memmove(dest: *mut u8, src: *const u8, n: usize) -> *mut u8;
    fn memset(dest: *mut u8, c: i32, n: usize) -> *mut u8;
}

// FIXME: The `*4` and `*8` variants should be defined as aliases.

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "aapcs" fn __aeabi_memcpy(dest: *mut u8, src: *const u8, n: usize) {
    memcpy(dest, src, n);
}
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "aapcs" fn __aeabi_memcpy4(dest: *mut u8, src: *const u8, n: usize) {
    memcpy(dest, src, n);
}
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "aapcs" fn __aeabi_memcpy8(dest: *mut u8, src: *const u8, n: usize) {
    memcpy(dest, src, n);
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "aapcs" fn __aeabi_memmove(dest: *mut u8, src: *const u8, n: usize) {
    memmove(dest, src, n);
}
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "aapcs" fn __aeabi_memmove4(dest: *mut u8, src: *const u8, n: usize) {
    memmove(dest, src, n);
}
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "aapcs" fn __aeabi_memmove8(dest: *mut u8, src: *const u8, n: usize) {
    memmove(dest, src, n);
}

// Note the different argument order
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "aapcs" fn __aeabi_memset(dest: *mut u8, n: usize, c: i32) {
    memset(dest, c, n);
}
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "aapcs" fn __aeabi_memset4(dest: *mut u8, n: usize, c: i32) {
    memset(dest, c, n);
}
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "aapcs" fn __aeabi_memset8(dest: *mut u8, n: usize, c: i32) {
    memset(dest, c, n);
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "aapcs" fn __aeabi_memclr(dest: *mut u8, n: usize) {
    memset(dest, 0, n);
}
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "aapcs" fn __aeabi_memclr4(dest: *mut u8, n: usize) {
    memset(dest, 0, n);
}
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "aapcs" fn __aeabi_memclr8(dest: *mut u8, n: usize) {
    memset(dest, 0, n);
}
