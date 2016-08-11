use core::intrinsics;

// NOTE This function and the one below are implemented using assembly because they using a custom
// calling convention which can't be implemented using a normal Rust function
// TODO use `global_asm!`
#[naked]
#[no_mangle]
pub unsafe extern "aapcs" fn __aeabi_uidivmod() {
    asm!("push    { lr }
          sub     sp, sp, #4
          mov     r2, sp
          bl      __udivmodsi4
          ldr     r1, [sp]
          add     sp, sp, #4
          pop     { pc }");
    intrinsics::unreachable();
}

// TODO use `global_asm!`
#[naked]
#[no_mangle]
pub unsafe extern "aapcs" fn __aeabi_uldivmod() {
    asm!("push	{r11, lr}
          sub	sp, sp, #16
          add	r12, sp, #8
          str	r12, [sp]
          bl	__udivmoddi4
          ldr	r2, [sp, #8]
          ldr	r3, [sp, #12]
          add	sp, sp, #16
          pop	{r11, pc}");
    intrinsics::unreachable();
}

extern "C" {
    fn memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8;
    fn memmove(dest: *mut u8, src: *const u8, n: usize) -> *mut u8;
    fn memset(dest: *mut u8, c: i32, n: usize) -> *mut u8;
}

// FIXME: The `*4` and `*8` variants should be defined as aliases.

#[no_mangle]
pub unsafe extern "aapcs" fn __aeabi_memcpy(dest: *mut u8, src: *const u8, n: usize) {
    memcpy(dest, src, n);
}
#[no_mangle]
pub unsafe extern "aapcs" fn __aeabi_memcpy4(dest: *mut u8, src: *const u8, n: usize) {
    memcpy(dest, src, n);
}
#[no_mangle]
pub unsafe extern "aapcs" fn __aeabi_memcpy8(dest: *mut u8, src: *const u8, n: usize) {
    memcpy(dest, src, n);
}

#[no_mangle]
pub unsafe extern "aapcs" fn __aeabi_memmove(dest: *mut u8, src: *const u8, n: usize) {
    memmove(dest, src, n);
}
#[no_mangle]
pub unsafe extern "aapcs" fn __aeabi_memmove4(dest: *mut u8, src: *const u8, n: usize) {
    memmove(dest, src, n);
}
#[no_mangle]
pub unsafe extern "aapcs" fn __aeabi_memmove8(dest: *mut u8, src: *const u8, n: usize) {
    memmove(dest, src, n);
}

// Note the different argument order
#[no_mangle]
pub unsafe extern "aapcs" fn __aeabi_memset(dest: *mut u8, n: usize, c: i32) {
    memset(dest, c, n);
}
#[no_mangle]
pub unsafe extern "aapcs" fn __aeabi_memset4(dest: *mut u8, n: usize, c: i32) {
    memset(dest, c, n);
}
#[no_mangle]
pub unsafe extern "aapcs" fn __aeabi_memset8(dest: *mut u8, n: usize, c: i32) {
    memset(dest, c, n);
}

#[no_mangle]
pub unsafe extern "aapcs" fn __aeabi_memclr(dest: *mut u8, n: usize) {
    memset(dest, 0, n);
}
#[no_mangle]
pub unsafe extern "aapcs" fn __aeabi_memclr4(dest: *mut u8, n: usize) {
    memset(dest, 0, n);
}
#[no_mangle]
pub unsafe extern "aapcs" fn __aeabi_memclr8(dest: *mut u8, n: usize) {
    memset(dest, 0, n);
}
