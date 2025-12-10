/// Halt the CPU until the next interrupt.
#[inline(always)]
#[stable(feature = "v810_intrinsics", since = "1.94.0")]
pub fn halt() {
    unsafe { crate::arch::asm!("halt") };
}

/// Enable interrupts.
#[inline(always)]
#[cfg(target_feature = "nintendo")]
#[stable(feature = "v810_intrinsics", since = "1.94.0")]
pub fn cli() {
    unsafe { crate::arch::asm!("cli", options(nomem)) };
}

/// Disable interrupts.
#[inline(always)]
#[cfg(target_feature = "nintendo")]
#[stable(feature = "v810_intrinsics", since = "1.94.0")]
pub fn sei() {
    unsafe { crate::arch::asm!("sei", options(nomem)) };
}
