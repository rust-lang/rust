//! This test exercises the combined effect of the `cfg(target_has_atomic_equal_alignment = "...")`
//! implementation in the compiler plus usage of said `cfg(target_has_atomic_equal_alignment)` in
//! `core` for the `Atomic64::from_mut` API.
//!
//! This test is a basic smoke test: that `AtomicU64::from_mut` is gated by
//! `#[cfg(target_has_atomic_equal_alignment = "8")]`, which is only available on platforms where
//! `AtomicU64` has the same alignment as `u64`. This is notably *not* satisfied by `x86_32`, where
//! they have differing alignments. Thus, `AtomicU64::from_mut` should *not* be available on
//! `x86_32` linux and should report assoc item not found, if the `cfg` is working correctly.
//! Conversely, `AtomicU64::from_mut` *should* be available on `x86_64` linux where the alignment
//! matches.

//@ revisions: alignment_mismatch alignment_matches

// This should fail on 32-bit x86 linux...
//@[alignment_mismatch] only-x86
//@[alignment_mismatch] only-linux

// ... but pass on 64-bit x86_64 linux.
//@[alignment_matches] only-x86_64
//@[alignment_matches] only-linux

fn main() {
    core::sync::atomic::AtomicU64::from_mut(&mut 0u64);
    //[alignment_mismatch]~^ ERROR no function or associated item named `from_mut` found for struct `AtomicU64`
    //[alignment_matches]~^^ ERROR use of unstable library feature `atomic_from_mut`
}
