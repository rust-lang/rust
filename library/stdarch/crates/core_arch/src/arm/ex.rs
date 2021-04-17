// Reference: Section 5.4.4 "LDREX / STREX" of ACLE

/// Removes the exclusive lock created by LDREX
// Supported: v6, v6K, v7-M, v7-A, v7-R
// Not supported: v5, v6-M
// NOTE: there's no dedicated CLREX instruction in v6 (<v6k); to clear the exclusive monitor users
// have to do a dummy STREX operation
#[cfg(any(
    all(target_feature = "v6k", not(target_feature = "mclass")), // excludes v6-M
    all(target_feature = "v7", target_feature = "mclass"), // v7-M
    doc
))]
pub unsafe fn __clrex() {
    extern "C" {
        #[link_name = "llvm.arm.clrex"]
        fn clrex();
    }

    clrex()
}

/// Executes a exclusive LDR instruction for 8 bit value.
// Supported: v6K, v7-M, v7-A, v7-R
// Not supported: v5, v6, v6-M
#[cfg(any(
    target_feature = "v6k", // includes v7-M but excludes v6-M
    doc
))]
pub unsafe fn __ldrexb(p: *const u8) -> u8 {
    extern "C" {
        #[link_name = "llvm.arm.ldrex.p0i8"]
        fn ldrex8(p: *const u8) -> u32;
    }

    ldrex8(p) as u8
}

/// Executes a exclusive LDR instruction for 16 bit value.
// Supported: v6K, v7-M, v7-A, v7-R, v8
// Not supported: v5, v6, v6-M
#[cfg(any(
    target_feature = "v6k", // includes v7-M but excludes v6-M
    doc
))]
pub unsafe fn __ldrexh(p: *const u16) -> u16 {
    extern "C" {
        #[link_name = "llvm.arm.ldrex.p0i16"]
        fn ldrex16(p: *const u16) -> u32;
    }

    ldrex16(p) as u16
}

/// Executes a exclusive LDR instruction for 32 bit value.
// Supported: v6, v7-M, v6K, v7-A, v7-R, v8
// Not supported: v5, v6-M
#[cfg(any(
    all(target_feature = "v6", not(target_feature = "mclass")), // excludes v6-M
    all(target_feature = "v7", target_feature = "mclass"), // v7-M
    doc
))]
pub unsafe fn __ldrex(p: *const u32) -> u32 {
    extern "C" {
        #[link_name = "llvm.arm.ldrex.p0i32"]
        fn ldrex32(p: *const u32) -> u32;
    }

    ldrex32(p)
}

/// Executes a exclusive STR instruction for 8 bit values
///
/// Returns `0` if the operation succeeded, or `1` if it failed
// supported: v6K, v7-M, v7-A, v7-R
// Not supported: v5, v6, v6-M
#[cfg(any(
    target_feature = "v6k", // includes v7-M but excludes v6-M
    doc
))]
pub unsafe fn __strexb(value: u32, addr: *mut u8) -> u32 {
    extern "C" {
        #[link_name = "llvm.arm.strex.p0i8"]
        fn strex8(value: u32, addr: *mut u8) -> u32;
    }

    strex8(value, addr)
}

/// Executes a exclusive STR instruction for 16 bit values
///
/// Returns `0` if the operation succeeded, or `1` if it failed
// Supported: v6K, v7-M, v7-A, v7-R, v8
// Not supported: v5, v6, v6-M
#[cfg(target_feature = "aarch64")]
#[cfg(any(
    target_feature = "v6k", // includes v7-M but excludes v6-M
    doc
))]
pub unsafe fn __strexh(value: u16, addr: *mut u16) -> u32 {
    extern "C" {
        #[link_name = "llvm.arm.strex.p0i16"]
        fn strex16(value: u32, addr: *mut u16) -> u32;
    }

    strex16(value as u32, addr)
}

/// Executes a exclusive STR instruction for 32 bit values
///
/// Returns `0` if the operation succeeded, or `1` if it failed
// Supported: v6, v7-M, v6K, v7-A, v7-R, v8
// Not supported: v5, v6-M
#[cfg(any(
    all(target_feature = "v6", not(target_feature = "mclass")), // excludes v6-M
    all(target_feature = "v7", target_feature = "mclass"), // v7-M
    doc
))]
pub unsafe fn __strex(value: u32, addr: *mut u32) -> u32 {
    extern "C" {
        #[link_name = "llvm.arm.strex.p0i32"]
        fn strex32(value: u32, addr: *mut u32) -> u32;
    }

    strex32(value, addr)
}
