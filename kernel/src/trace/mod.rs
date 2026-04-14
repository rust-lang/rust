//! Kernel tracing and timing utilities.
//!
//! All timing in the kernel derives from the `BootRuntimeBase::mono_ticks()` source.

pub mod irq_ring;

/// Returns the current monotonic time in nanoseconds.
///
/// This is the single authoritative time source for all kernel-internal
/// timing, profiling, and tracing. It uses the runtime's `mono_ticks()`
/// which is converted to nanoseconds.
///
/// Returns 0 if the runtime is not yet initialized.
#[inline]
pub fn now() -> u64 {
    // Access runtime_base directly instead of using a registered function pointer.
    // This ensures we use the same timebase as the syscall handlers.
    let rt = crate::runtime_base();
    let ticks = rt.mono_ticks();
    let freq = rt.mono_freq_hz();
    if freq == 0 {
        return 0;
    }
    // Convert ticks to nanoseconds: ns = ticks * 1_000_000_000 / freq
    // Use u128 to avoid overflow for large tick counts
    (ticks as u128 * 1_000_000_000 / freq as u128) as u64
}

/// Returns the current monotonic time in nanoseconds, or 0 if runtime unavailable.
///
/// This is a non-panicking version that can be called very early in boot.
#[inline]
pub fn now_or_zero() -> u64 {
    now()
}
