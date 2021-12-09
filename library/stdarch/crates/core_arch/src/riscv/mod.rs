//! RISC-V intrinsics

/// Generates the `PAUSE` instruction
///
/// The PAUSE instruction is a HINT that indicates the current hart's rate of instruction retirement
/// should be temporarily reduced or paused. The duration of its effect must be bounded and may be zero.
#[inline]
pub fn pause() {
    unsafe { crate::arch::asm!(".word 0x0100000F", options(nomem, nostack)) }
}
