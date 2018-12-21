use core::intrinsics;

// NOTE This function and the ones below are implemented using assembly because they using a custom
// calling convention which can't be implemented using a normal Rust function.
// NOTE The only difference between the iOS and non-iOS versions of those functions is that the iOS
// versions use 3 leading underscores in the names of called functions instead of 2.
#[cfg(not(any(target_os = "ios", target_env = "msvc")))]
#[naked]
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
pub unsafe fn __aeabi_uidivmod() {
    asm!("push {lr}
          sub sp, sp, #4
          mov r2, sp
          bl __udivmodsi4
          ldr r1, [sp]
          add sp, sp, #4
          pop {pc}" ::: "memory" : "volatile");
    intrinsics::unreachable();
}

#[cfg(target_os = "ios")]
#[naked]
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
pub unsafe fn __aeabi_uidivmod() {
    asm!("push {lr}
          sub sp, sp, #4
          mov r2, sp
          bl ___udivmodsi4
          ldr r1, [sp]
          add sp, sp, #4
          pop {pc}" ::: "memory" : "volatile");
    intrinsics::unreachable();
}

#[cfg(not(target_os = "ios"))]
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
          pop {r4, pc}" ::: "memory" : "volatile");
    intrinsics::unreachable();
}

#[cfg(target_os = "ios")]
#[naked]
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
pub unsafe fn __aeabi_uldivmod() {
    asm!("push {r4, lr}
          sub sp, sp, #16
          add r4, sp, #8
          str r4, [sp]
          bl ___udivmoddi4
          ldr r2, [sp, #8]
          ldr r3, [sp, #12]
          add sp, sp, #16
          pop {r4, pc}" ::: "memory" : "volatile");
    intrinsics::unreachable();
}

#[cfg(not(target_os = "ios"))]
#[naked]
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
pub unsafe fn __aeabi_idivmod() {
    asm!("push {r0, r1, r4, lr}
          bl __aeabi_idiv
          pop {r1, r2}
          muls r2, r2, r0
          subs r1, r1, r2
          pop {r4, pc}" ::: "memory" : "volatile");
    intrinsics::unreachable();
}

#[cfg(target_os = "ios")]
#[naked]
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
pub unsafe fn __aeabi_idivmod() {
    asm!("push {r0, r1, r4, lr}
          bl ___aeabi_idiv
          pop {r1, r2}
          muls r2, r2, r0
          subs r1, r1, r2
          pop {r4, pc}" ::: "memory" : "volatile");
    intrinsics::unreachable();
}

#[cfg(not(target_os = "ios"))]
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
          pop {r4, pc}" ::: "memory" : "volatile");
    intrinsics::unreachable();
}

#[cfg(target_os = "ios")]
#[naked]
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
pub unsafe fn __aeabi_ldivmod() {
    asm!("push {r4, lr}
          sub sp, sp, #16
          add r4, sp, #8
          str r4, [sp]
          bl ___divmoddi4
          ldr r2, [sp, #8]
          ldr r3, [sp, #12]
          add sp, sp, #16
          pop {r4, pc}" ::: "memory" : "volatile");
    intrinsics::unreachable();
}

// FIXME: The `*4` and `*8` variants should be defined as aliases.

#[cfg(not(target_os = "ios"))]
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[cfg_attr(thumb, linkage = "weak")]
pub unsafe extern "aapcs" fn __aeabi_memcpy(dest: *mut u8, src: *const u8, n: usize) {
    ::mem::memcpy(dest, src, n);
}

#[cfg(not(target_os = "ios"))]
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[cfg_attr(thumb, linkage = "weak")]
pub unsafe extern "aapcs" fn __aeabi_memcpy4(dest: *mut u8, src: *const u8, mut n: usize) {
    use core::ptr;

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

#[cfg(not(target_os = "ios"))]
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[cfg_attr(thumb, linkage = "weak")]
pub unsafe extern "aapcs" fn __aeabi_memcpy8(dest: *mut u8, src: *const u8, n: usize) {
    __aeabi_memcpy4(dest, src, n);
}

#[cfg(not(target_os = "ios"))]
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[cfg_attr(thumb, linkage = "weak")]
pub unsafe extern "aapcs" fn __aeabi_memmove(dest: *mut u8, src: *const u8, n: usize) {
    ::mem::memmove(dest, src, n);
}

#[cfg(not(target_os = "ios"))]
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[cfg_attr(thumb, linkage = "weak")]
pub unsafe extern "aapcs" fn __aeabi_memmove4(dest: *mut u8, src: *const u8, n: usize) {
    __aeabi_memmove(dest, src, n);
}

#[cfg(not(target_os = "ios"))]
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[cfg_attr(thumb, linkage = "weak")]
pub unsafe extern "aapcs" fn __aeabi_memmove8(dest: *mut u8, src: *const u8, n: usize) {
    __aeabi_memmove(dest, src, n);
}

#[cfg(not(target_os = "ios"))]
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[cfg_attr(thumb, linkage = "weak")]
pub unsafe extern "aapcs" fn __aeabi_memset(dest: *mut u8, n: usize, c: i32) {
    // Note the different argument order
    ::mem::memset(dest, c, n);
}

#[cfg(not(target_os = "ios"))]
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[cfg_attr(thumb, linkage = "weak")]
pub unsafe extern "aapcs" fn __aeabi_memset4(dest: *mut u8, mut n: usize, c: i32) {
    use core::ptr;

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

#[cfg(not(target_os = "ios"))]
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[cfg_attr(thumb, linkage = "weak")]
pub unsafe extern "aapcs" fn __aeabi_memset8(dest: *mut u8, n: usize, c: i32) {
    __aeabi_memset4(dest, n, c);
}

#[cfg(not(target_os = "ios"))]
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[cfg_attr(thumb, linkage = "weak")]
pub unsafe extern "aapcs" fn __aeabi_memclr(dest: *mut u8, n: usize) {
    __aeabi_memset(dest, n, 0);
}

#[cfg(not(target_os = "ios"))]
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[cfg_attr(thumb, linkage = "weak")]
pub unsafe extern "aapcs" fn __aeabi_memclr4(dest: *mut u8, n: usize) {
    __aeabi_memset4(dest, n, 0);
}

#[cfg(not(target_os = "ios"))]
#[cfg_attr(not(feature = "mangled-names"), no_mangle)]
#[cfg_attr(thumb, linkage = "weak")]
pub unsafe extern "aapcs" fn __aeabi_memclr8(dest: *mut u8, n: usize) {
    __aeabi_memset4(dest, n, 0);
}

#[no_mangle]
#[cfg(any(target_pointer_width = "16", target_pointer_width = "32", target_pointer_width = "64"))]
pub extern "C" fn __clzsi2(mut x: usize) -> usize {
  // TODO: const this? Requires const if
  let mut y: usize;
  let mut n: usize = {
    #[cfg(target_pointer_width = "64")]
    {
      64
    }
    #[cfg(target_pointer_width = "32")]
    {
      32
    }
    #[cfg(target_pointer_width = "16")]
    {
      16
    }
  };
  #[cfg(target_pointer_width = "64")]
  {
    y = x >> 32;
    if y != 0 {
      n -= 32;
      x = y;
    }
  }
  #[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
  {
    y = x >> 16;
    if y != 0 {
      n -= 16;
      x = y;
    }
  }
  y = x >> 8;
  if y != 0 {
    n -= 8;
    x = y;
  }
  y = x >> 4;
  if y != 0 {
    n -= 4;
    x = y;
  }
  y = x >> 2;
  if y != 0 {
    n -= 2;
    x = y;
  }
  y = x >> 1;
  if y != 0 {
    n - 2
  } else {
    n - x
  }
}
