use core::sync::atomic::{AtomicU32, Ordering};
use core::{arch, mem};

// Kernel-provided user-mode helper functions:
// https://www.kernel.org/doc/Documentation/arm/kernel_user_helpers.txt
unsafe fn __kuser_cmpxchg(oldval: u32, newval: u32, ptr: *mut u32) -> bool {
    // FIXME(volatile): the third parameter is a volatile pointer
    // SAFETY: kernel docs specify a known address with the given signature
    let f = unsafe {
        mem::transmute::<_, extern "C" fn(u32, u32, *mut u32) -> u32>(0xffff0fc0usize as *const ())
    };
    f(oldval, newval, ptr) == 0
}

unsafe fn __kuser_memory_barrier() {
    // SAFETY: kernel docs specify a known address with the given signature
    let f = unsafe { mem::transmute::<_, extern "C" fn()>(0xffff0fa0usize as *const ()) };
    f();
}

// Word-align a pointer
fn align_ptr<T>(ptr: *mut T) -> *mut u32 {
    // This gives us a mask of 0 when T == u32 since the pointer is already
    // supposed to be aligned, which avoids any masking in that case.
    let ptr_mask = 3 & (4 - mem::size_of::<T>());
    (ptr as usize & !ptr_mask) as *mut u32
}

// Calculate the shift and mask of a value inside an aligned word
fn get_shift_mask<T>(ptr: *mut T) -> (u32, u32) {
    // Mask to get the low byte/halfword/word
    let mask = match mem::size_of::<T>() {
        1 => 0xff,
        2 => 0xffff,
        4 => 0xffffffff,
        _ => unreachable!(),
    };

    // If we are on big-endian then we need to adjust the shift accordingly
    let endian_adjust = if cfg!(target_endian = "little") {
        0
    } else {
        4 - mem::size_of::<T>() as u32
    };

    // Shift to get the desired element in the word
    let ptr_mask = 3 & (4 - mem::size_of::<T>());
    let shift = ((ptr as usize & ptr_mask) as u32 ^ endian_adjust) * 8;

    (shift, mask)
}

// Extract a value from an aligned word
fn extract_aligned(aligned: u32, shift: u32, mask: u32) -> u32 {
    (aligned >> shift) & mask
}

// Insert a value into an aligned word
fn insert_aligned(aligned: u32, val: u32, shift: u32, mask: u32) -> u32 {
    (aligned & !(mask << shift)) | ((val & mask) << shift)
}

/// Performs a relaxed atomic load of 4 bytes at `ptr`. Some of the bytes are allowed to be out of
/// bounds as long as `size_of::<T>()` bytes are in bounds.
///
/// # Safety
///
/// - `ptr` must be 4-aligned.
/// - `size_of::<T>()` must be at most 4.
/// - if `size_of::<T>() == 1`, `ptr` or `ptr` offset by 1, 2 or 3 bytes must be valid for a relaxed
///   atomic read of 1 byte.
/// - if `size_of::<T>() == 2`, `ptr` or `ptr` offset by 2 bytes must be valid for a relaxed atomic
///   read of 2 bytes.
/// - if `size_of::<T>() == 4`, `ptr` must be valid for a relaxed atomic read of 4 bytes.
// FIXME: assert some of the preconditions in debug mode
unsafe fn atomic_load_aligned<T>(ptr: *mut u32) -> u32 {
    const { assert!(size_of::<T>() <= 4) };
    if size_of::<T>() == 4 {
        // SAFETY: As `T` has a size of 4, the caller garantees this is sound.
        unsafe { AtomicU32::from_ptr(ptr).load(Ordering::Relaxed) }
    } else {
        // SAFETY:
        // As all 4 bytes pointed to by `ptr` might not be dereferenceable due to being out of
        // bounds when doing atomic operations on a `u8`/`i8`/`u16`/`i16`, inline ASM is used to
        // avoid causing undefined behaviour. However, as `ptr` is 4-aligned and at least 1 byte of
        // `ptr` is dereferencable, the load won't cause a segfault as the page size is always
        // larger than 4 bytes.
        // The `ldr` instruction does not touch the stack or flags, or write to memory, so
        // `nostack`, `preserves_flags` and `readonly` are sound. The caller garantees that `ptr` is
        // 4-aligned, as required by `ldr`.
        unsafe {
            let res: u32;
            arch::asm!(
                "ldr {res}, [{ptr}]",
                ptr = in(reg) ptr,
                res = lateout(reg) res,
                options(nostack, preserves_flags, readonly)
            );
            res
        }
    }
}

// Generic atomic read-modify-write operation
unsafe fn atomic_rmw<T, F: Fn(u32) -> u32, G: Fn(u32, u32) -> u32>(ptr: *mut T, f: F, g: G) -> u32 {
    let aligned_ptr = align_ptr(ptr);
    let (shift, mask) = get_shift_mask(ptr);

    loop {
        // FIXME(safety): preconditions review needed
        let curval_aligned = unsafe { atomic_load_aligned::<T>(aligned_ptr) };
        let curval = extract_aligned(curval_aligned, shift, mask);
        let newval = f(curval);
        let newval_aligned = insert_aligned(curval_aligned, newval, shift, mask);
        // FIXME(safety): preconditions review needed
        if unsafe { __kuser_cmpxchg(curval_aligned, newval_aligned, aligned_ptr) } {
            return g(curval, newval);
        }
    }
}

// Generic atomic compare-exchange operation
unsafe fn atomic_cmpxchg<T>(ptr: *mut T, oldval: u32, newval: u32) -> u32 {
    let aligned_ptr = align_ptr(ptr);
    let (shift, mask) = get_shift_mask(ptr);

    loop {
        // SAFETY: the caller must guarantee that the pointer is valid for read and write
        // and aligned to the element size.
        let curval_aligned = unsafe { atomic_load_aligned::<T>(aligned_ptr) };
        let curval = extract_aligned(curval_aligned, shift, mask);
        if curval != oldval {
            return curval;
        }
        let newval_aligned = insert_aligned(curval_aligned, newval, shift, mask);
        // SAFETY: the caller must guarantee that the pointer is valid for read and write
        // and aligned to the element size.
        if unsafe { __kuser_cmpxchg(curval_aligned, newval_aligned, aligned_ptr) } {
            return oldval;
        }
    }
}

macro_rules! atomic_rmw {
    ($name:ident, $ty:ty, $op:expr, $fetch:expr) => {
        intrinsics! {
            pub unsafe extern "C" fn $name(ptr: *mut $ty, val: $ty) -> $ty {
                // SAFETY: the caller must guarantee that the pointer is valid for read and write
                // and aligned to the element size.
                unsafe {
                    atomic_rmw(
                        ptr,
                        |x| $op(x as $ty, val) as u32,
                        |old, new| $fetch(old, new)
                    ) as $ty
                }
            }
        }
    };

    (@old $name:ident, $ty:ty, $op:expr) => {
        atomic_rmw!($name, $ty, $op, |old, _| old);
    };

    (@new $name:ident, $ty:ty, $op:expr) => {
        atomic_rmw!($name, $ty, $op, |_, new| new);
    };
}
macro_rules! atomic_cmpxchg {
    ($name:ident, $ty:ty) => {
        intrinsics! {
            pub unsafe extern "C" fn $name(ptr: *mut $ty, oldval: $ty, newval: $ty) -> $ty {
                // SAFETY: the caller must guarantee that the pointer is valid for read and write
                // and aligned to the element size.
                unsafe { atomic_cmpxchg(ptr, oldval as u32, newval as u32) as $ty }
            }
        }
    };
}

include!("arm_thumb_shared.rs");

intrinsics! {
    pub unsafe extern "C" fn __sync_synchronize() {
       // SAFETY: preconditions are the same as the calling function.
       unsafe {  __kuser_memory_barrier() };
    }
}
