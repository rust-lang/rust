use core::intrinsics;

// NOTE This function and the ones below are implemented using assembly because they using a custom
// calling convention which can't be implemented using a normal Rust function
#[naked]
#[cfg_attr(not(test), no_mangle)]
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
#[cfg_attr(not(test), no_mangle)]
pub extern "C" fn __aeabi_dadd(a: f64, b: f64) -> f64 {
    ::float::add::__adddf3(a, b)
}

#[cfg_attr(not(test), no_mangle)]
pub extern "C" fn __aeabi_fadd(a: f32, b: f32) -> f32 {
    ::float::add::__addsf3(a, b)
}

#[cfg_attr(not(test), no_mangle)]
pub extern "C" fn __aeabi_idiv(a: i32, b: i32) -> i32 {
    ::int::sdiv::__divsi3(a, b)
}

#[cfg_attr(not(test), no_mangle)]
pub extern "C" fn __aeabi_lasr(a: i64, b: u32) -> i64 {
    ::int::shift::__ashrdi3(a, b)
}

#[cfg_attr(not(test), no_mangle)]
pub extern "C" fn __aeabi_llsl(a: u64, b: u32) -> u64 {
    ::int::shift::__ashldi3(a, b)
}

#[cfg_attr(not(test), no_mangle)]
pub extern "C" fn __aeabi_llsr(a: u64, b: u32) -> u64 {
    ::int::shift::__lshrdi3(a, b)
}

#[cfg_attr(not(test), no_mangle)]
pub extern "C" fn __aeabi_lmul(a: u64, b: u64) -> u64 {
    ::int::mul::__muldi3(a, b)
}

#[cfg_attr(not(test), no_mangle)]
pub extern "C" fn __aeabi_uidiv(a: u32, b: u32) -> u32 {
    ::int::udiv::__udivsi3(a, b)
}

extern "C" {
    fn memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8;
    fn memmove(dest: *mut u8, src: *const u8, n: usize) -> *mut u8;
    fn memset(dest: *mut u8, c: i32, n: usize) -> *mut u8;
}

// FIXME: The `*4` and `*8` variants should be defined as aliases.

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn __aeabi_memcpy(dest: *mut u8, src: *const u8, n: usize) {
    memcpy(dest, src, n);
}
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn __aeabi_memcpy4(dest: *mut u8, src: *const u8, n: usize) {
    memcpy(dest, src, n);
}
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn __aeabi_memcpy8(dest: *mut u8, src: *const u8, n: usize) {
    memcpy(dest, src, n);
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn __aeabi_memmove(dest: *mut u8, src: *const u8, n: usize) {
    memmove(dest, src, n);
}
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn __aeabi_memmove4(dest: *mut u8, src: *const u8, n: usize) {
    memmove(dest, src, n);
}
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn __aeabi_memmove8(dest: *mut u8, src: *const u8, n: usize) {
    memmove(dest, src, n);
}

// Note the different argument order
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn __aeabi_memset(dest: *mut u8, n: usize, c: i32) {
    memset(dest, c, n);
}
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn __aeabi_memset4(dest: *mut u8, n: usize, c: i32) {
    memset(dest, c, n);
}
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn __aeabi_memset8(dest: *mut u8, n: usize, c: i32) {
    memset(dest, c, n);
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn __aeabi_memclr(dest: *mut u8, n: usize) {
    memset(dest, 0, n);
}
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn __aeabi_memclr4(dest: *mut u8, n: usize) {
    memset(dest, 0, n);
}
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn __aeabi_memclr8(dest: *mut u8, n: usize) {
    memset(dest, 0, n);
}


#[cfg(test)]
mod tests {
    use quickcheck::TestResult;
    use qc::{U32, U64};

    quickcheck!{
        fn uldivmod(n: U64, d: U64) -> TestResult {
            let (n, d) = (n.0, d.0);
            if d == 0 {
                TestResult::discard()
            } else {
                let q: u64;
                let r: u64;
                unsafe {
                    // The inline asm is a bit tricky here, LLVM will allocate
                    // both r0 and r1 when we specify a 64-bit value for {r0}.
                    asm!("bl __aeabi_uldivmod"
                         : "={r0}" (q), "={r2}" (r)
                         : "{r0}" (n), "{r2}" (d)
                         : "r12", "lr", "flags");
                }
                TestResult::from_bool(q == n / d && r == n % d)
            }
        }

        fn uidivmod(n: U32, d: U32) -> TestResult {
            let (n, d) = (n.0, d.0);
            if d == 0 {
                TestResult::discard()
            } else {
                let q: u32;
                let r: u32;
                unsafe {
                    asm!("bl __aeabi_uidivmod"
                         : "={r0}" (q), "={r1}" (r)
                         : "{r0}" (n), "{r1}" (d)
                         : "r2", "r3", "r12", "lr", "flags");
                }
                TestResult::from_bool(q == n / d && r == n % d)
            }
        }

        fn ldivmod(n: U64, d: U64) -> TestResult {
            let (n, d) = (n.0 as i64, d.0 as i64);
            if d == 0 {
                TestResult::discard()
            } else {
                let q: i64;
                let r: i64;
                unsafe {
                    // The inline asm is a bit tricky here, LLVM will allocate
                    // both r0 and r1 when we specify a 64-bit value for {r0}.
                    asm!("bl __aeabi_ldivmod"
                         : "={r0}" (q), "={r2}" (r)
                         : "{r0}" (n), "{r2}" (d)
                         : "r12", "lr", "flags");
                }
                TestResult::from_bool(q == n / d && r == n % d)
            }
        }

        fn idivmod(n: U32, d: U32) -> TestResult {
            let (n, d) = (n.0 as i32, d.0 as i32);
            if d == 0 {
                TestResult::discard()
            } else {
                let q: i32;
                let r: i32;
                unsafe {
                    asm!("bl __aeabi_idivmod"
                         : "={r0}" (q), "={r1}" (r)
                         : "{r0}" (n), "{r1}" (d)
                         : "r2", "r3", "r12", "lr", "flags");
                }
                TestResult::from_bool(q == n / d && r == n % d)
            }
        }
    }
}
