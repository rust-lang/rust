//! SVE-accelerated `Vec::retain_mut` implementation.
//!
//! Two-phase algorithm per 64-element chunk:
//! - Phase A (scalar): evaluates predicate exactly once, in order, into a bool mask.
//! - Phase B (SVE): uses `COMPACT` instruction to compress retained elements.

use core::{cmp, mem, ptr};

use super::Vec;
use crate::alloc::Allocator;

const CHUNK_SIZE: usize = 64;

pub(super) const MIN_SVE_SIZE_1: usize = 32;
pub(super) const MIN_SVE_SIZE_2: usize = 32;
pub(super) const MIN_SVE_SIZE_4: usize = 64;
pub(super) const MIN_SVE_SIZE_8: usize = 64;

/// Guard for the SVE retain path. On panic, this guard:
/// 1. Scalar-compresses the already-decided prefix of the current chunk.
/// 2. Copies the untouched tail forward.
/// 3. Sets the Vec length.
struct PanicGuard<'a, T, A: Allocator> {
    v: &'a mut Vec<T, A>,
    /// Start index of the current chunk within the Vec.
    read: usize,
    /// Write cursor (accumulated from previous chunks).
    write: usize,
    /// How many elements in the current chunk have been decided by the predicate.
    decided: usize,

    mask: &'a mut [bool; CHUNK_SIZE],
    /// Original length of the Vec.
    original_len: usize,
}

impl<T, A: Allocator> Drop for PanicGuard<'_, T, A> {
    #[cold]
    fn drop(&mut self) {
        // Scalar-compress the decided prefix: move kept elements to write position.
        let mut dst = self.write;
        for i in 0..self.decided {
            if self.mask[i] {
                // SAFETY: read + i < original_len (in-bounds).
                let src = unsafe { self.v.as_ptr().add(self.read + i) };
                // SAFETY: Continued from above.
                let dst_ptr = unsafe { self.v.as_mut_ptr().add(dst) };
                // SAFETY: src and dst_ptr < original_len
                unsafe { ptr::copy(src, dst_ptr, 1) };
                dst += 1;
            }
        }

        // Copy the untouched tail.
        let untouched_start = self.read + self.decided;
        let untouched_len = self.original_len - untouched_start;
        if untouched_len > 0 {
            // SAFETY: untouched_start..original_len are valid; dst + untouched_len <= original_len.
            unsafe {
                ptr::copy(
                    self.v.as_ptr().add(untouched_start),
                    self.v.as_mut_ptr().add(dst),
                    untouched_len,
                );
            }
            dst += untouched_len;
        }

        // SAFETY: After filling holes, all items are in contiguous memory.
        unsafe { self.v.set_len(dst) };
    }
}

/// # Safety
///
/// - `size_of::<T>() == 1/2/4/8`
/// - Called on aarch64 + SVE target.
pub(crate) unsafe fn chunked_retain<T, F, A: Allocator>(v: &mut Vec<T, A>, mut f: F)
where
    F: FnMut(&mut T) -> bool,
{
    let original_len = v.len();
    let mut guard = PanicGuard {
        v,
        read: 0,
        write: 0,
        decided: 0,
        mask: &mut [false; CHUNK_SIZE],
        original_len,
    };

    while guard.read < guard.original_len {
        let chunk_len = cmp::min(CHUNK_SIZE, guard.original_len - guard.read);

        guard.decided = 0;

        // Phase A: scalar predicate evaluation (exactly once, in order).
        for i in 0..chunk_len {
            // SAFETY: read + i < original_len
            let cur = unsafe { &mut *guard.v.as_mut_ptr().add(guard.read + i) };
            guard.mask[i] = f(cur);
            guard.decided = i + 1;
            if !guard.mask[i] {
                // SAFETY: read + i < original_len, and after marking `mask` and `decided`,
                // the guard can properly handles the case where drop_in_place panics.
                unsafe { ptr::drop_in_place(cur) };
            }
        }

        // Phase B: SVE compress.
        let kept = match mem::size_of::<T>() {
            // SAFETY: write <= read and the dispatch guarantees size_of::<T>() matches
            // the kernel lane width.
            1 => unsafe {
                compact8_kernel(
                    guard.v.as_mut_ptr().add(guard.read),
                    guard.v.as_mut_ptr().add(guard.write),
                    guard.mask.as_ptr(),
                    chunk_len,
                )
            },
            // SAFETY: Same as above.
            2 => unsafe {
                compact16_kernel(
                    guard.v.as_mut_ptr().add(guard.read),
                    guard.v.as_mut_ptr().add(guard.write),
                    guard.mask.as_ptr(),
                    chunk_len,
                )
            },
            // SAFETY: Same as above.
            4 => unsafe {
                compact32_kernel(
                    guard.v.as_mut_ptr().add(guard.read),
                    guard.v.as_mut_ptr().add(guard.write),
                    guard.mask.as_ptr(),
                    chunk_len,
                )
            },
            // SAFETY: Same as above.
            8 => unsafe {
                compact64_kernel(
                    guard.v.as_mut_ptr().add(guard.read),
                    guard.v.as_mut_ptr().add(guard.write),
                    guard.mask.as_ptr(),
                    chunk_len,
                )
            },
            _ => unreachable!(),
        };

        guard.write += kept;
        guard.read += chunk_len;
    }

    // SAFETY: write <= original_len, all retained elements are packed at the front.
    unsafe { guard.v.set_len(guard.write) };
    mem::forget(guard);
}

macro_rules! sve_compact_kernel {
    (
        $name:ident,
        size = $size:literal,
        lane = $lane:literal,
        mem_lane = $mem_lane:literal,
        shift = $shift:literal,
        inc = $inc:literal
    ) => {
        /// SVE compress kernel: pack retained elements from `src` to `dst`
        /// according to `mask`. Returns the number of retained elements.
        ///
        /// # Safety
        ///
        /// - `src` is valid for `chunk_len` reads of `T`, `dst` for `chunk_len`
        ///   writes, `mask` for `chunk_len` bool reads.
        /// - `size_of::<T>()` equals this kernel's lane width in bytes.
        #[target_feature(enable = "sve")]
        #[inline]
        unsafe fn $name<T>(
            src: *const T,
            dst: *mut T,
            mask: *const bool,
            chunk_len: usize,
        ) -> usize {
            debug_assert_eq!(mem::size_of::<T>(), $size);

            let idx_in = 0usize;
            let mut idx_out = 0usize;
            // SVE intrinsics require treating the data as integers, which doesn't
            // correctly handle provenance and uninitialized padding.
            //
            // SAFETY: whilelo predicates every load/store to the remaining elements.
            unsafe {
                core::arch::asm!(
                    concat!("whilelo p0.", $lane, ", xzr, {len}"),
                    "2:",
                    concat!("ld1b {{ z0.", $lane, " }}, p0/z, [{mask}, {idx_in}]"),
                    concat!("cmpne p1.", $lane, ", p0/z, z0.", $lane, ", #0"),
                    concat!("ld1", $mem_lane, " {{ z1.", $lane, " }}, p1/z, [{src}, {idx_in}", $shift, "]"),
                    concat!("compact z1.", $lane, ", p1, z1.", $lane),
                    concat!("cntp {kept}, p0, p1.", $lane),
                    concat!("whilelo p2.", $lane, ", xzr, {kept}"),
                    concat!("st1", $mem_lane, " {{ z1.", $lane, " }}, p2, [{dst}, {idx_out}", $shift, "]"),
                    concat!("add {idx_out}, {idx_out}, {kept}"),
                    concat!("inc", $inc, " {idx_in}"),
                    concat!("whilelo p0.", $lane, ", {idx_in}, {len}"),
                    "b.first 2b",
                    src = in(reg) src,
                    dst = in(reg) dst,
                    mask = in(reg) mask,
                    len = in(reg) chunk_len,
                    idx_in = inout(reg) idx_in => _,
                    idx_out = inout(reg) idx_out,
                    kept = out(reg) _,
                    out("p0") _,
                    out("p1") _,
                    out("p2") _,
                    out("z0") _,
                    out("z1") _,
                    options(nostack),
                );
            }
            idx_out
        }
    };
}

sve_compact_kernel!(compact8_kernel, size = 1, lane = "s", mem_lane = "b", shift = "", inc = "w");

sve_compact_kernel!(
    compact16_kernel,
    size = 2,
    lane = "s",
    mem_lane = "h",
    shift = ", lsl #1",
    inc = "w"
);

sve_compact_kernel!(
    compact32_kernel,
    size = 4,
    lane = "s",
    mem_lane = "w",
    shift = ", lsl #2",
    inc = "w"
);

sve_compact_kernel!(
    compact64_kernel,
    size = 8,
    lane = "d",
    mem_lane = "d",
    shift = ", lsl #3",
    inc = "d"
);
