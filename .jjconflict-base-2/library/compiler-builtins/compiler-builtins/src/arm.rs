// Interfaces used by naked trampolines.
// SAFETY: these are defined in compiler-builtins
unsafe extern "C" {
    fn __udivmodsi4(a: u32, b: u32, rem: *mut u32) -> u32;
    fn __udivmoddi4(a: u64, b: u64, rem: *mut u64) -> u64;
    fn __divmoddi4(a: i64, b: i64, rem: *mut i64) -> i64;
}

// SAFETY: these are defined in compiler-builtins
unsafe extern "custom" {
    // AAPCS is not always the correct ABI for these intrinsics, but we only use this to
    // forward another `__aeabi_` call so it doesn't matter.
    fn __aeabi_idiv();
}

intrinsics! {
    // NOTE This function and the ones below are implemented using assembly because they are using a
    // custom calling convention which can't be implemented using a normal Rust function.
    #[unsafe(naked)]
    #[cfg(not(target_env = "msvc"))]
    pub unsafe extern "custom" fn __aeabi_uidivmod() {
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
    pub unsafe extern "custom" fn __aeabi_uldivmod() {
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
    pub unsafe extern "custom" fn __aeabi_idivmod() {
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
    pub unsafe extern "custom" fn __aeabi_ldivmod() {
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
        debug_assert!(dst.addr().is_multiple_of(8));
        debug_assert!(src.addr().is_multiple_of(8));

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
        debug_assert!(dst.addr().is_multiple_of(4));
        debug_assert!(src.addr().is_multiple_of(4));

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
        debug_assert!(dst.addr().is_multiple_of(8));
        debug_assert!(src.addr().is_multiple_of(8));

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
        debug_assert!(dst.addr().is_multiple_of(8));

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
        debug_assert!(dst.addr().is_multiple_of(4));

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
        debug_assert!(dst.addr().is_multiple_of(8));

        // SAFETY: memclr preconditions apply, less strict alignment.
        unsafe { __aeabi_memset4(dst, n, 0) };
    }

    // =================================
    // Unaligned memory access functions
    // see https://github.com/ARM-software/abi-aa/blob/main/rtabi32/rtabi32.rst#533unaligned-memory-access

    // Read a `u32` from a possibly unaligned address.
    //
    // # Safety
    //
    // `address` must be valid for reading four bytes.
    #[unsafe(naked)]
    #[cfg(not(any(target_vendor = "apple", target_env = "msvc")))]
    pub unsafe extern "aapcs" fn __aeabi_uread4(address: *const u8) -> u32 {
        core::cfg_select! {
            all(thumb1_only, target_endian = "little") => {
                core::arch::naked_asm!(
                    "ldrb r1, [r0]",
                    "ldrb r2, [r0, #1]",
                    "lsls r2, r2, #8",
                    "adds r1, r2, r1",
                    "ldrb r2, [r0, #2]",
                    "lsls r2, r2, #16",
                    "ldrb r0, [r0, #3]",
                    "lsls r0, r0, #24",
                    "adds r0, r0, r2",
                    "adds r0, r0, r1",
                    "bx lr",
                );
            }
            all(thumb1_only, target_endian = "big") => {
                core::arch::naked_asm!(
                    "ldrb r1, [r0, #3]",
                    "ldrb r2, [r0, #2]",
                    "lsls r2, r2, #8",
                    "adds r1, r2, r1",
                    "ldrb r2, [r0, #1]",
                    "lsls r2, r2, #16",
                    "ldrb r0, [r0]",
                    "lsls r0, r0, #24",
                    "adds r0, r0, r2",
                    "adds r0, r0, r1",
                    "bx lr",
                );
            }
            all(not(thumb1_only), target_endian = "little") => {
                core::arch::naked_asm!(
                    "ldrb r1, [r0]",
                    "ldrb r2, [r0, #1]",
                    "ldrb r3, [r0, #2]",
                    "ldrb r0, [r0, #3]",
                    "orr r0, r3, r0, lsl #8",
                    "orr r1, r1, r2, lsl #8",
                    "orr r0, r1, r0, lsl #16",
                    "bx lr",
                );
            }
            all(not(thumb1_only), target_endian = "big") => {
                core::arch::naked_asm!(
                    "ldrb r1, [r0]",
                    "ldrb r2, [r0, #1]",
                    "ldrb r3, [r0, #2]",
                    "ldrb r0, [r0, #3]",
                    "orr r1, r2, r1, lsl #8",
                    "orr r0, r0, r3, lsl #8",
                    "orr r0, r0, r1, lsl #16",
                    "bx lr",
                );
            }
        }
    }

    // Read a `u64` from a possibly unaligned address.
    //
    // # Safety
    //
    // `address` must be valid for reading eight bytes.
    #[unsafe(naked)]
    #[cfg(not(any(target_vendor = "apple", target_env = "msvc")))]
    pub unsafe extern "aapcs" fn __aeabi_uread8(address: *const u8) -> u64 {
        core::cfg_select! {
            all(thumb1_only, target_endian = "little") => {
                core::arch::naked_asm!(
                    "ldrb r1, [r0]",
                    "ldrb r2, [r0, #1]",
                    "lsls r2, r2, #8",
                    "adds r1, r2, r1",
                    "ldrb r2, [r0, #2]",
                    "lsls r2, r2, #16",
                    "ldrb r3, [r0, #3]",
                    "lsls r3, r3, #24",
                    "adds r2, r3, r2",
                    "adds r2, r2, r1",
                    "ldrb r1, [r0, #4]",
                    "ldrb r3, [r0, #5]",
                    "lsls r3, r3, #8",
                    "adds r1, r3, r1",
                    "ldrb r3, [r0, #6]",
                    "lsls r3, r3, #16",
                    "ldrb r0, [r0, #7]",
                    "lsls r0, r0, #24",
                    "adds r0, r0, r3",
                    "adds r1, r0, r1",
                    "movs r0, r2",
                    "bx lr",
                );
            }
            all(thumb1_only, target_endian = "big") => {
                core::arch::naked_asm!(
                    "ldrb r1, [r0, #3]",
                    "ldrb r2, [r0, #2]",
                    "lsls r2, r2, #8",
                    "adds r1, r2, r1",
                    "ldrb r2, [r0, #1]",
                    "lsls r2, r2, #16",
                    "ldrb r3, [r0]",
                    "lsls r3, r3, #24",
                    "adds r2, r3, r2",
                    "adds r2, r2, r1",
                    "ldrb r1, [r0, #7]",
                    "ldrb r3, [r0, #6]",
                    "lsls r3, r3, #8",
                    "adds r1, r3, r1",
                    "ldrb r3, [r0, #5]",
                    "lsls r3, r3, #16",
                    "ldrb r0, [r0, #4]",
                    "lsls r0, r0, #24",
                    "adds r0, r0, r3",
                    "adds r1, r0, r1",
                    "movs r0, r2",
                    "bx lr",
                );
            }
            all(not(thumb1_only), target_endian = "little") => {
                core::arch::naked_asm!(
                    "ldrb r12, [r0]",
                    "ldrb r2, [r0, #1]",
                    "ldrb r3, [r0, #2]",
                    "ldrb r1, [r0, #3]",
                    "orr r1, r3, r1, lsl #8",
                    "orr r2, r12, r2, lsl #8",
                    "orr r2, r2, r1, lsl #16",
                    "ldrb r1, [r0, #5]",
                    "ldrb r3, [r0, #4]!",
                    "orr r1, r3, r1, lsl #8",
                    "ldrb r3, [r0, #2]",
                    "ldrb r0, [r0, #3]",
                    "orr r0, r3, r0, lsl #8",
                    "orr r1, r1, r0, lsl #16",
                    "mov r0, r2",
                    "bx lr",
                );
            }
            all(not(thumb1_only), target_endian = "big") => {
                core::arch::naked_asm!(
                    "ldrb r12, [r0]",
                    "ldrb r2, [r0, #1]",
                    "ldrb r3, [r0, #2]",
                    "ldrb r1, [r0, #3]",
                    "orr r2, r2, r12, lsl #8",
                    "orr r1, r1, r3, lsl #8",
                    "orr r2, r1, r2, lsl #16",
                    "mov r1, r0",
                    "ldrb r0, [r0, #5]",
                    "ldrb r3, [r1, #4]!",
                    "orr r0, r0, r3, lsl #8",
                    "ldrb r3, [r1, #2]",
                    "ldrb r1, [r1, #3]",
                    "orr r1, r1, r3, lsl #8",
                    "orr r1, r1, r0, lsl #16",
                    "mov r0, r2",
                    "bx lr",
                );
            }
        }
    }

    // Write a `u32` to a possibly unaligned address, returning the value written.
    //
    // # Safety
    //
    // `address` must be valid for writing four bytes.
    #[unsafe(naked)]
    #[cfg(not(any(target_vendor = "apple", target_env = "msvc")))]
    pub unsafe extern "aapcs" fn __aeabi_uwrite4(value: u32, address: *mut u8) -> u32 {
        core::cfg_select! {
            target_endian = "little" => {
                core::arch::naked_asm!(
                    "lsrs r2, r0, #24",
                    "strb r0, [r1]",
                    "strb r2, [r1, #3]",
                    "lsrs r2, r0, #16",
                    "strb r2, [r1, #2]",
                    "lsrs r2, r0, #8",
                    "strb r2, [r1, #1]",
                    "bx lr",
                );
            }
            target_endian = "big" => {
                core::arch::naked_asm!(
                    "lsrs r2, r0, #8",
                    "strb r0, [r1, #3]",
                    "strb r2, [r1, #2]",
                    "lsrs r2, r0, #16",
                    "strb r2, [r1, #1]",
                    "lsrs r2, r0, #24",
                    "strb r2, [r1]",
                    "bx lr",
                );
            }
        }
    }

    // Write a `u64` to a possibly unaligned address, returning the value written.
    //
    // # Safety
    //
    // `address` must be valid for writing eight bytes.
    #[unsafe(naked)]
    #[cfg(not(any(target_vendor = "apple", target_env = "msvc")))]
    pub unsafe extern "aapcs" fn __aeabi_uwrite8(value: u64, address: *mut u8) -> u64 {
        core::cfg_select! {
            target_endian = "little" => {
                core::arch::naked_asm!(
                    "strb r0, [r2, #0]",
                    "lsrs r3, r0, #8",
                    "strb r3, [r2, #1]",
                    "lsrs r3, r0, #16",
                    "strb r3, [r2, #2]",
                    "lsrs r3, r0, #24",
                    "strb r3, [r2, #3]",
                    "strb r1, [r2, #4]",
                    "lsrs r3, r1, #8",
                    "strb r3, [r2, #5]",
                    "lsrs r3, r1, #16",
                    "strb r3, [r2, #6]",
                    "lsrs r3, r1, #24",
                    "strb r3, [r2, #7]",
                    "bx lr",
                );
            }
            target_endian = "big" => {
                core::arch::naked_asm!(
                    "lsrs r3, r0, #24",
                    "strb r3, [r2, #0]",
                    "lsrs r3, r0, #16",
                    "strb r3, [r2, #1]",
                    "lsrs r3, r0, #8",
                    "strb r3, [r2, #2]",
                    "strb r0, [r2, #3]",
                    "lsrs r3, r1, #24",
                    "strb r3, [r2, #4]",
                    "lsrs r3, r1, #16",
                    "strb r3, [r2, #5]",
                    "lsrs r3, r1, #8",
                    "strb r3, [r2, #6]",
                    "strb r1, [r2, #7]",
                    "bx lr",
                );
            }
        }
    }
}
