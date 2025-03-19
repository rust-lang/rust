#![cfg(not(feature = "no-asm"))]

// Interfaces used by naked trampolines.
extern "C" {
    fn __udivmodsi4(a: u32, b: u32, rem: *mut u32) -> u32;
    fn __udivmoddi4(a: u64, b: u64, rem: *mut u64) -> u64;
    fn __divmoddi4(a: i64, b: i64, rem: *mut i64) -> i64;
}

extern "aapcs" {
    // AAPCS is not always the correct ABI for these intrinsics, but we only use this to
    // forward another `__aeabi_` call so it doesn't matter.
    fn __aeabi_idiv(a: i32, b: i32) -> i32;
}

intrinsics! {
    // NOTE This function and the ones below are implemented using assembly because they are using a
    // custom calling convention which can't be implemented using a normal Rust function.
    #[unsafe(naked)]
    #[cfg(not(target_env = "msvc"))]
    pub unsafe extern "C" fn __aeabi_uidivmod() {
        core::arch::naked_asm!(
            "push {{lr}}",
            "sub sp, sp, #4",
            "mov r2, sp",
            "bl {trampoline}",
            "ldr r1, [sp]",
            "add sp, sp, #4",
            "pop {{pc}}",
            trampoline = sym crate::arm::__udivmodsi4
        );
    }

    #[unsafe(naked)]
    pub unsafe extern "C" fn __aeabi_uldivmod() {
        core::arch::naked_asm!(
            "push {{r4, lr}}",
            "sub sp, sp, #16",
            "add r4, sp, #8",
            "str r4, [sp]",
            "bl {trampoline}",
            "ldr r2, [sp, #8]",
            "ldr r3, [sp, #12]",
            "add sp, sp, #16",
            "pop {{r4, pc}}",
            trampoline = sym crate::arm::__udivmoddi4
        );
    }

    #[unsafe(naked)]
    pub unsafe extern "C" fn __aeabi_idivmod() {
        core::arch::naked_asm!(
            "push {{r0, r1, r4, lr}}",
            "bl {trampoline}",
            "pop {{r1, r2}}",
            "muls r2, r2, r0",
            "subs r1, r1, r2",
            "pop {{r4, pc}}",
            trampoline = sym crate::arm::__aeabi_idiv,
        );
    }

    #[unsafe(naked)]
    pub unsafe extern "C" fn __aeabi_ldivmod() {
        core::arch::naked_asm!(
            "push {{r4, lr}}",
            "sub sp, sp, #16",
            "add r4, sp, #8",
            "str r4, [sp]",
            "bl {trampoline}",
            "ldr r2, [sp, #8]",
            "ldr r3, [sp, #12]",
            "add sp, sp, #16",
            "pop {{r4, pc}}",
            trampoline = sym crate::arm::__divmoddi4,
        );
    }

    // FIXME(arm): The `*4` and `*8` variants should be defined as aliases.

    /// `memcpy` provided with the `aapcs` ABI.
    ///
    /// # Safety
    ///
    /// Usual `memcpy` requirements apply.
    #[cfg(not(target_vendor = "apple"))]
    pub unsafe extern "aapcs" fn __aeabi_memcpy(dst: *mut u8, src: *const u8, n: usize) {
        // SAFETY: memcpy preconditions apply.
        unsafe { crate::mem::memcpy(dst, src, n) };
    }

    /// `memcpy` for 4-byte alignment.
    ///
    /// # Safety
    ///
    /// Usual `memcpy` requirements apply. Additionally, `dest` and `src` must be aligned to
    /// four bytes.
    #[cfg(not(target_vendor = "apple"))]
    pub unsafe extern "aapcs" fn __aeabi_memcpy4(dst: *mut u8, src: *const u8, n: usize) {
        // We are guaranteed 4-alignment, so accessing at u32 is okay.
        let mut dst = dst.cast::<u32>();
        let mut src = src.cast::<u32>();
        debug_assert!(dst.is_aligned());
        debug_assert!(src.is_aligned());
        let mut n = n;

        while n >= 4 {
            // SAFETY: `dst` and `src` are both valid for at least 4 bytes, from
            // `memcpy` preconditions and the loop guard.
            unsafe { *dst = *src };

            // FIXME(addr): if we can make this end-of-address-space safe without losing
            // performance, we may want to consider that.
            // SAFETY: memcpy is not expected to work at the end of the address space
            unsafe {
                dst = dst.offset(1);
                src = src.offset(1);
            }

            n -= 4;
        }

        // SAFETY: `dst` and `src` will still be valid for `n` bytes
        unsafe { __aeabi_memcpy(dst.cast::<u8>(), src.cast::<u8>(), n) };
    }

    /// `memcpy` for 8-byte alignment.
    ///
    /// # Safety
    ///
    /// Usual `memcpy` requirements apply. Additionally, `dest` and `src` must be aligned to
    /// eight bytes.
    #[cfg(not(target_vendor = "apple"))]
    pub unsafe extern "aapcs" fn __aeabi_memcpy8(dst: *mut u8, src: *const u8, n: usize) {
        debug_assert!(dst.addr() & 7 == 0);
        debug_assert!(src.addr() & 7 == 0);

        // SAFETY: memcpy preconditions apply, less strict alignment.
        unsafe { __aeabi_memcpy4(dst, src, n) };
    }

    /// `memmove` provided with the `aapcs` ABI.
    ///
    /// # Safety
    ///
    /// Usual `memmove` requirements apply.
    #[cfg(not(target_vendor = "apple"))]
    pub unsafe extern "aapcs" fn __aeabi_memmove(dst: *mut u8, src: *const u8, n: usize) {
        // SAFETY: memmove preconditions apply.
        unsafe { crate::mem::memmove(dst, src, n) };
    }

    /// `memmove` for 4-byte alignment.
    ///
    /// # Safety
    ///
    /// Usual `memmove` requirements apply. Additionally, `dest` and `src` must be aligned to
    /// four bytes.
    #[cfg(not(any(target_vendor = "apple", target_env = "msvc")))]
    pub unsafe extern "aapcs" fn __aeabi_memmove4(dst: *mut u8, src: *const u8, n: usize) {
        debug_assert!(dst.addr() & 3 == 0);
        debug_assert!(src.addr() & 3 == 0);

        // SAFETY: same preconditions, less strict aligment.
        unsafe { __aeabi_memmove(dst, src, n) };
    }

    /// `memmove` for 8-byte alignment.
    ///
    /// # Safety
    ///
    /// Usual `memmove` requirements apply. Additionally, `dst` and `src` must be aligned to
    /// eight bytes.
    #[cfg(not(any(target_vendor = "apple", target_env = "msvc")))]
    pub unsafe extern "aapcs" fn __aeabi_memmove8(dst: *mut u8, src: *const u8, n: usize) {
        debug_assert!(dst.addr() & 7 == 0);
        debug_assert!(src.addr() & 7 == 0);

        // SAFETY: memmove preconditions apply, less strict alignment.
        unsafe { __aeabi_memmove(dst, src, n) };
    }

    /// `memset` provided with the `aapcs` ABI.
    ///
    /// # Safety
    ///
    /// Usual `memset` requirements apply.
    #[cfg(not(target_vendor = "apple"))]
    pub unsafe extern "aapcs" fn __aeabi_memset(dst: *mut u8, n: usize, c: i32) {
        // Note the different argument order
        // SAFETY: memset preconditions apply.
        unsafe { crate::mem::memset(dst, c, n) };
    }

    /// `memset` for 4-byte alignment.
    ///
    /// # Safety
    ///
    /// Usual `memset` requirements apply. Additionally, `dest` and `src` must be aligned to
    /// four bytes.
    #[cfg(not(target_vendor = "apple"))]
    pub unsafe extern "aapcs" fn __aeabi_memset4(dst: *mut u8, n: usize, c: i32) {
        let mut dst = dst.cast::<u32>();
        debug_assert!(dst.is_aligned());
        let mut n = n;

        let byte = (c as u32) & 0xff;
        let c = (byte << 24) | (byte << 16) | (byte << 8) | byte;

        while n >= 4 {
            // SAFETY: `dst` is valid for at least 4 bytes, from `memset` preconditions and
            // the loop guard.
            unsafe { *dst = c };

            // FIXME(addr): if we can make this end-of-address-space safe without losing
            // performance, we may want to consider that.
            // SAFETY: memcpy is not expected to work at the end of the address space
            unsafe {
                dst = dst.offset(1);
            }
            n -= 4;
        }

        // SAFETY: `dst` will still be valid for `n` bytes
        unsafe { __aeabi_memset(dst.cast::<u8>(), n, byte as i32) };
    }

    /// `memset` for 8-byte alignment.
    ///
    /// # Safety
    ///
    /// Usual `memset` requirements apply. Additionally, `dst` and `src` must be aligned to
    /// eight bytes.
    #[cfg(not(target_vendor = "apple"))]
    pub unsafe extern "aapcs" fn __aeabi_memset8(dst: *mut u8, n: usize, c: i32) {
        debug_assert!(dst.addr() & 7 == 0);

        // SAFETY: memset preconditions apply, less strict alignment.
        unsafe { __aeabi_memset4(dst, n, c) };
    }

    /// `memclr` provided with the `aapcs` ABI.
    ///
    /// # Safety
    ///
    /// Usual `memclr` requirements apply.
    #[cfg(not(target_vendor = "apple"))]
    pub unsafe extern "aapcs" fn __aeabi_memclr(dst: *mut u8, n: usize) {
        // SAFETY: memclr preconditions apply, less strict alignment.
        unsafe { __aeabi_memset(dst, n, 0) };
    }

    /// `memclr` for 4-byte alignment.
    ///
    /// # Safety
    ///
    /// Usual `memclr` requirements apply. Additionally, `dest` and `src` must be aligned to
    /// four bytes.
    #[cfg(not(any(target_vendor = "apple", target_env = "msvc")))]
    pub unsafe extern "aapcs" fn __aeabi_memclr4(dst: *mut u8, n: usize) {
        debug_assert!(dst.addr() & 3 == 0);

        // SAFETY: memclr preconditions apply, less strict alignment.
        unsafe { __aeabi_memset4(dst, n, 0) };
    }

    /// `memclr` for 8-byte alignment.
    ///
    /// # Safety
    ///
    /// Usual `memclr` requirements apply. Additionally, `dst` and `src` must be aligned to
    /// eight bytes.
    #[cfg(not(any(target_vendor = "apple", target_env = "msvc")))]
    pub unsafe extern "aapcs" fn __aeabi_memclr8(dst: *mut u8, n: usize) {
        debug_assert!(dst.addr() & 7 == 0);

        // SAFETY: memclr preconditions apply, less strict alignment.
        unsafe { __aeabi_memset4(dst, n, 0) };
    }
}
