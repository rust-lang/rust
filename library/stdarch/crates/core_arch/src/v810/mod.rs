/// Halt the CPU until the next interrupt.
#[inline(always)]
pub fn halt() {
    unsafe { crate::arch::asm!("halt") };
}

/// Enable interrupts.
#[inline(always)]
#[cfg(target_feature = "nintendo")]
pub fn cli() {
    unsafe { crate::arch::asm!("cli", options(nomem)) };
}

/// Disable interrupts.
#[inline(always)]
#[cfg(target_feature = "nintendo")]
pub fn sei() {
    unsafe { crate::arch::asm!("sei", options(nomem)) };
}
