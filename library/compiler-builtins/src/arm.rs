use core::{intrinsics, ptr};

use mem;

// NOTE This function and the ones below are implemented using assembly because they using a custom
// calling convention which can't be implemented using a normal Rust function
#[naked]
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
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
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
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
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
pub unsafe fn __aeabi_idivmod() {
    asm!("push {r0, r1, r4, lr}
          bl __aeabi_idiv
          pop {r1, r2}
          muls r2, r2, r0
          subs r1, r1, r2
          pop {r4, pc}");
    intrinsics::unreachable();
}

#[naked]
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
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

// FIXME: The `*4` and `*8` variants should be defined as aliases.

#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[linkage = "weak"]
pub unsafe extern "aapcs" fn __aeabi_memcpy(dest: *mut u8, src: *const u8, n: usize) {
    mem::memcpy(dest, src, n);
}

#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[linkage = "weak"]
pub unsafe extern "aapcs" fn __aeabi_memcpy4(dest: *mut u8, src: *const u8, mut n: usize) {
    let mut dest = dest as *mut u32;
    let mut src = src as *mut u32;

    while n >= 4 {
        ptr::write(dest, ptr::read(src));
        dest = dest.offset(1);
        src = src.offset(1);
        n -= 4;
    }

    __aeabi_memcpy(dest as *mut u8, src as *const u8, n);
}

#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[linkage = "weak"]
pub unsafe extern "aapcs" fn __aeabi_memcpy8(dest: *mut u8, src: *const u8, n: usize) {
    __aeabi_memcpy4(dest, src, n);
}

#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[linkage = "weak"]
pub unsafe extern "aapcs" fn __aeabi_memmove(dest: *mut u8, src: *const u8, n: usize) {
    mem::memmove(dest, src, n);
}

#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[linkage = "weak"]
pub unsafe extern "aapcs" fn __aeabi_memmove4(dest: *mut u8, src: *const u8, n: usize) {
    __aeabi_memmove(dest, src, n);
}

#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[linkage = "weak"]
pub unsafe extern "aapcs" fn __aeabi_memmove8(dest: *mut u8, src: *const u8, n: usize) {
    __aeabi_memmove(dest, src, n);
}

// Note the different argument order
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[linkage = "weak"]
pub unsafe extern "aapcs" fn __aeabi_memset(dest: *mut u8, n: usize, c: i32) {
    mem::memset(dest, c, n);
}

#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[linkage = "weak"]
pub unsafe extern "aapcs" fn __aeabi_memset4(dest: *mut u8, mut n: usize, c: i32) {
    let mut dest = dest as *mut u32;

    let byte = (c as u32) & 0xff;
    let c = (byte << 24) | (byte << 16) | (byte << 8) | byte;

    while n >= 4 {
        ptr::write(dest, c);
        dest = dest.offset(1);
        n -= 4;
    }

    __aeabi_memset(dest as *mut u8, n, byte as i32);
}

#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[linkage = "weak"]
pub unsafe extern "aapcs" fn __aeabi_memset8(dest: *mut u8, n: usize, c: i32) {
    __aeabi_memset4(dest, n, c);
}

#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[linkage = "weak"]
pub unsafe extern "aapcs" fn __aeabi_memclr(dest: *mut u8, n: usize) {
    __aeabi_memset(dest, n, 0);
}

#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[linkage = "weak"]
pub unsafe extern "aapcs" fn __aeabi_memclr4(dest: *mut u8, n: usize) {
    __aeabi_memset4(dest, n, 0);
}

#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[linkage = "weak"]
pub unsafe extern "aapcs" fn __aeabi_memclr8(dest: *mut u8, n: usize) {
    __aeabi_memset4(dest, n, 0);
}
