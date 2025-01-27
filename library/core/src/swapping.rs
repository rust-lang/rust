use crate::{hint, intrinsics, mem, ptr};

//#[rustc_const_stable_indirect]
//#[rustc_allow_const_fn_unstable(const_eval_select)]
#[rustc_const_unstable(feature = "const_swap_nonoverlapping", issue = "133668")]
#[inline]
pub(crate) const unsafe fn swap_nonoverlapping<T>(x: *mut T, y: *mut T, count: usize) {
    intrinsics::const_eval_select!(
        @capture[T] { x: *mut T, y: *mut T, count: usize }:
        if const {
            // At compile-time we want to always copy this in chunks of `T`, to ensure that if there
            // are pointers inside `T` we will copy them in one go rather than trying to copy a part
            // of a pointer (which would not work).
            // SAFETY: Same preconditions as this function
            unsafe { swap_nonoverlapping_const(x, y, count) }
        } else {
            // At runtime we want to make sure not to swap byte-for-byte for types like [u8; 15],
            // and swapping as `MaybeUninit<T>` doesn't actually work as untyped for things like
            // T = (u16, u8), so we type-erase to raw bytes and swap that way.
            // SAFETY: Same preconditions as this function
            unsafe { swap_nonoverlapping_runtime(x, y, count) }
        }
    )
}

/// Same behavior and safety conditions as [`swap_nonoverlapping`]
#[rustc_const_stable_indirect]
#[inline]
const unsafe fn swap_nonoverlapping_const<T>(x: *mut T, y: *mut T, count: usize) {
    let x = x.cast::<mem::MaybeUninit<T>>();
    let y = y.cast::<mem::MaybeUninit<T>>();
    let mut i = 0;
    while i < count {
        // SAFETY: By precondition, `i` is in-bounds because it's below `n`
        // and because the two input ranges are non-overlapping and read/writeable,
        // these individual items inside them are too.
        unsafe {
            intrinsics::untyped_swap_nonoverlapping::<T>(x.add(i), y.add(i));
        }

        i += 1;
    }
}

// Scale the monomorphizations with the size of the machine, roughly.
const MAX_ALIGN: usize = align_of::<usize>().pow(2);

/// Same behavior and safety conditions as [`swap_nonoverlapping`]
#[inline]
unsafe fn swap_nonoverlapping_runtime<T>(x: *mut T, y: *mut T, count: usize) {
    let bytes = {
        let slice = ptr::slice_from_raw_parts(x, count);
        // SAFETY: Because they both exist in memory and don't overlap, they
        // must be legal slice sizes (below `isize::MAX` bytes).
        unsafe { mem::size_of_val_raw(slice) }
    };

    // Generating *untyped* loops for every type is silly, so we polymorphize away
    // the actual type, but we want to take advantage of alignment if possible,
    // so monomorphize for a restricted set of possible alignments.
    macro_rules! delegate_by_alignment {
        ($($p:pat => $align:expr,)+) => {{
            #![allow(unreachable_patterns)]
            match const { align_of::<T>() } {
                $(
                    $p => {
                        swap_nonoverlapping_bytes::<$align>(x.cast(), y.cast(), bytes);
                    }
                )+
            }
        }};
    }

    // SAFETY:
    unsafe {
        delegate_by_alignment! {
            MAX_ALIGN.. => MAX_ALIGN,
            64.. => 64,
            32.. => 32,
            16.. => 16,
            8.. => 8,
            4.. => 4,
            2.. => 2,
            _ => 1,
        }
    }
}

/// # Safety:
/// - `x` and `y` must be aligned to `ALIGN`
/// - `bytes` must be a multiple of `ALIGN`
/// - They must be readable, writable, and non-overlapping for `bytes` bytes
#[inline]
unsafe fn swap_nonoverlapping_bytes<const ALIGN: usize>(
    x: *mut mem::MaybeUninit<u8>,
    y: *mut mem::MaybeUninit<u8>,
    bytes: usize,
) {
    // SAFETY: Two legal non-overlapping regions can't be bigger than this.
    // (And they couldn't have made allocations any bigger either anyway.)
    // FIXME: Would be nice to have a type for this instead of the assume.
    unsafe { hint::assert_unchecked(bytes < isize::MAX as usize) };

    let mut i = 0;
    macro_rules! swap_next_n {
        ($n:expr) => {{
            let x: *mut mem::MaybeUninit<[u8; $n]> = x.add(i).cast();
            let y: *mut mem::MaybeUninit<[u8; $n]> = y.add(i).cast();
            swap_nonoverlapping_aligned_chunk::<ALIGN, [u8; $n]>(
                x.as_mut_unchecked(),
                y.as_mut_unchecked(),
            );
            i += $n;
        }};
    }

    while bytes - i >= MAX_ALIGN {
        const { assert!(MAX_ALIGN >= ALIGN) };
        // SAFETY: the const-assert above confirms we're only ever called with
        // an alignment equal to or smaller than max align, so this is necessarily
        // aligned, and the while loop ensures there's enough read/write memory.
        unsafe {
            swap_next_n!(MAX_ALIGN);
        }
    }

    macro_rules! handle_tail {
        ($($n:literal)+) => {$(
            if const { $n % ALIGN == 0 } {
                // Checking this way simplifies the block end to just add+test,
                // rather than needing extra math before the check.
                if (bytes & $n) != 0 {
                    // SAFETY: The above swaps were bigger, so could not have
                    // impacted the `$n`-relevant bit, so checking `bytes & $n`
                    // was equivalent to `bytes - i >= $n`, and thus we have
                    // enough space left to swap another `$n` bytes.
                    unsafe {
                        swap_next_n!($n);
                    }
                }
            }
        )+};
    }
    const { assert!(MAX_ALIGN <= 64) };
    handle_tail!(32 16 8 4 2 1);

    debug_assert_eq!(i, bytes);
}

/// Swaps the `C` behind `x` and `y` as untyped memory
///
/// # Safety
///
/// Both `x` and `y` must be aligned to `ALIGN`, in addition to their normal alignment.
/// They must be readable and writeable for `sizeof(C)` bytes, as usual for `&mut`s.
///
/// (The actual instantiations are usually `C = [u8; _]`, so we get the alignment
/// information from the loads by `assume`ing the passed-in alignment.)
// Don't let MIR inline this, because we really want it to keep its noalias metadata
#[rustc_no_mir_inline]
#[inline]
unsafe fn swap_nonoverlapping_aligned_chunk<const ALIGN: usize, C>(
    x: &mut mem::MaybeUninit<C>,
    y: &mut mem::MaybeUninit<C>,
) {
    assert!(size_of::<C>() % ALIGN == 0);

    let x = ptr::from_mut(x);
    let y = ptr::from_mut(y);

    // SAFETY: One of our preconditions.
    unsafe {
        hint::assert_unchecked(x.is_aligned_to(ALIGN));
        hint::assert_unchecked(y.is_aligned_to(ALIGN));
    }

    // SAFETY: The memory is readable and writable because these were passed to
    // us as mutable references, and the untyped swap doesn't need validity.
    unsafe {
        intrinsics::untyped_swap_nonoverlapping::<C>(x, y);
    }
}
