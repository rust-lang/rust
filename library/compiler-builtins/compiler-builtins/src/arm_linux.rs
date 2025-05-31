use core::sync::atomic::{AtomicU32, Ordering};
use core::{arch, mem};

// Kernel-provided user-mode helper functions:
// https://www.kernel.org/doc/Documentation/arm/kernel_user_helpers.txt
unsafe fn __kuser_cmpxchg(oldval: u32, newval: u32, ptr: *mut u32) -> bool {
    let f: extern "C" fn(u32, u32, *mut u32) -> u32 = mem::transmute(0xffff0fc0usize as *const ());
    f(oldval, newval, ptr) == 0
}

unsafe fn __kuser_memory_barrier() {
    let f: extern "C" fn() = mem::transmute(0xffff0fa0usize as *const ());
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
unsafe fn atomic_load_aligned<T>(ptr: *mut u32) -> u32 {
    if mem::size_of::<T>() == 4 {
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
        let curval_aligned = atomic_load_aligned::<T>(aligned_ptr);
        let curval = extract_aligned(curval_aligned, shift, mask);
        let newval = f(curval);
        let newval_aligned = insert_aligned(curval_aligned, newval, shift, mask);
        if __kuser_cmpxchg(curval_aligned, newval_aligned, aligned_ptr) {
            return g(curval, newval);
        }
    }
}

// Generic atomic compare-exchange operation
unsafe fn atomic_cmpxchg<T>(ptr: *mut T, oldval: u32, newval: u32) -> u32 {
    let aligned_ptr = align_ptr(ptr);
    let (shift, mask) = get_shift_mask(ptr);

    loop {
        let curval_aligned = atomic_load_aligned::<T>(aligned_ptr);
        let curval = extract_aligned(curval_aligned, shift, mask);
        if curval != oldval {
            return curval;
        }
        let newval_aligned = insert_aligned(curval_aligned, newval, shift, mask);
        if __kuser_cmpxchg(curval_aligned, newval_aligned, aligned_ptr) {
            return oldval;
        }
    }
}

macro_rules! atomic_rmw {
    ($name:ident, $ty:ty, $op:expr, $fetch:expr) => {
        intrinsics! {
            pub unsafe extern "C" fn $name(ptr: *mut $ty, val: $ty) -> $ty {
                atomic_rmw(ptr, |x| $op(x as $ty, val) as u32, |old, new| $fetch(old, new)) as $ty
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
                atomic_cmpxchg(ptr, oldval as u32, newval as u32) as $ty
            }
        }
    };
}

atomic_rmw!(@old __sync_fetch_and_add_1, u8, |a: u8, b: u8| a.wrapping_add(b));
atomic_rmw!(@old __sync_fetch_and_add_2, u16, |a: u16, b: u16| a
    .wrapping_add(b));
atomic_rmw!(@old __sync_fetch_and_add_4, u32, |a: u32, b: u32| a
    .wrapping_add(b));

atomic_rmw!(@new __sync_add_and_fetch_1, u8, |a: u8, b: u8| a.wrapping_add(b));
atomic_rmw!(@new __sync_add_and_fetch_2, u16, |a: u16, b: u16| a
    .wrapping_add(b));
atomic_rmw!(@new __sync_add_and_fetch_4, u32, |a: u32, b: u32| a
    .wrapping_add(b));

atomic_rmw!(@old __sync_fetch_and_sub_1, u8, |a: u8, b: u8| a.wrapping_sub(b));
atomic_rmw!(@old __sync_fetch_and_sub_2, u16, |a: u16, b: u16| a
    .wrapping_sub(b));
atomic_rmw!(@old __sync_fetch_and_sub_4, u32, |a: u32, b: u32| a
    .wrapping_sub(b));

atomic_rmw!(@new __sync_sub_and_fetch_1, u8, |a: u8, b: u8| a.wrapping_sub(b));
atomic_rmw!(@new __sync_sub_and_fetch_2, u16, |a: u16, b: u16| a
    .wrapping_sub(b));
atomic_rmw!(@new __sync_sub_and_fetch_4, u32, |a: u32, b: u32| a
    .wrapping_sub(b));

atomic_rmw!(@old __sync_fetch_and_and_1, u8, |a: u8, b: u8| a & b);
atomic_rmw!(@old __sync_fetch_and_and_2, u16, |a: u16, b: u16| a & b);
atomic_rmw!(@old __sync_fetch_and_and_4, u32, |a: u32, b: u32| a & b);

atomic_rmw!(@new __sync_and_and_fetch_1, u8, |a: u8, b: u8| a & b);
atomic_rmw!(@new __sync_and_and_fetch_2, u16, |a: u16, b: u16| a & b);
atomic_rmw!(@new __sync_and_and_fetch_4, u32, |a: u32, b: u32| a & b);

atomic_rmw!(@old __sync_fetch_and_or_1, u8, |a: u8, b: u8| a | b);
atomic_rmw!(@old __sync_fetch_and_or_2, u16, |a: u16, b: u16| a | b);
atomic_rmw!(@old __sync_fetch_and_or_4, u32, |a: u32, b: u32| a | b);

atomic_rmw!(@new __sync_or_and_fetch_1, u8, |a: u8, b: u8| a | b);
atomic_rmw!(@new __sync_or_and_fetch_2, u16, |a: u16, b: u16| a | b);
atomic_rmw!(@new __sync_or_and_fetch_4, u32, |a: u32, b: u32| a | b);

atomic_rmw!(@old __sync_fetch_and_xor_1, u8, |a: u8, b: u8| a ^ b);
atomic_rmw!(@old __sync_fetch_and_xor_2, u16, |a: u16, b: u16| a ^ b);
atomic_rmw!(@old __sync_fetch_and_xor_4, u32, |a: u32, b: u32| a ^ b);

atomic_rmw!(@new __sync_xor_and_fetch_1, u8, |a: u8, b: u8| a ^ b);
atomic_rmw!(@new __sync_xor_and_fetch_2, u16, |a: u16, b: u16| a ^ b);
atomic_rmw!(@new __sync_xor_and_fetch_4, u32, |a: u32, b: u32| a ^ b);

atomic_rmw!(@old __sync_fetch_and_nand_1, u8, |a: u8, b: u8| !(a & b));
atomic_rmw!(@old __sync_fetch_and_nand_2, u16, |a: u16, b: u16| !(a & b));
atomic_rmw!(@old __sync_fetch_and_nand_4, u32, |a: u32, b: u32| !(a & b));

atomic_rmw!(@new __sync_nand_and_fetch_1, u8, |a: u8, b: u8| !(a & b));
atomic_rmw!(@new __sync_nand_and_fetch_2, u16, |a: u16, b: u16| !(a & b));
atomic_rmw!(@new __sync_nand_and_fetch_4, u32, |a: u32, b: u32| !(a & b));

atomic_rmw!(@old __sync_fetch_and_max_1, i8, |a: i8, b: i8| if a > b {
    a
} else {
    b
});
atomic_rmw!(@old __sync_fetch_and_max_2, i16, |a: i16, b: i16| if a > b {
    a
} else {
    b
});
atomic_rmw!(@old __sync_fetch_and_max_4, i32, |a: i32, b: i32| if a > b {
    a
} else {
    b
});

atomic_rmw!(@old __sync_fetch_and_umax_1, u8, |a: u8, b: u8| if a > b {
    a
} else {
    b
});
atomic_rmw!(@old __sync_fetch_and_umax_2, u16, |a: u16, b: u16| if a > b {
    a
} else {
    b
});
atomic_rmw!(@old __sync_fetch_and_umax_4, u32, |a: u32, b: u32| if a > b {
    a
} else {
    b
});

atomic_rmw!(@old __sync_fetch_and_min_1, i8, |a: i8, b: i8| if a < b {
    a
} else {
    b
});
atomic_rmw!(@old __sync_fetch_and_min_2, i16, |a: i16, b: i16| if a < b {
    a
} else {
    b
});
atomic_rmw!(@old __sync_fetch_and_min_4, i32, |a: i32, b: i32| if a < b {
    a
} else {
    b
});

atomic_rmw!(@old __sync_fetch_and_umin_1, u8, |a: u8, b: u8| if a < b {
    a
} else {
    b
});
atomic_rmw!(@old __sync_fetch_and_umin_2, u16, |a: u16, b: u16| if a < b {
    a
} else {
    b
});
atomic_rmw!(@old __sync_fetch_and_umin_4, u32, |a: u32, b: u32| if a < b {
    a
} else {
    b
});

atomic_rmw!(@old __sync_lock_test_and_set_1, u8, |_: u8, b: u8| b);
atomic_rmw!(@old __sync_lock_test_and_set_2, u16, |_: u16, b: u16| b);
atomic_rmw!(@old __sync_lock_test_and_set_4, u32, |_: u32, b: u32| b);

atomic_cmpxchg!(__sync_val_compare_and_swap_1, u8);
atomic_cmpxchg!(__sync_val_compare_and_swap_2, u16);
atomic_cmpxchg!(__sync_val_compare_and_swap_4, u32);

intrinsics! {
    pub unsafe extern "C" fn __sync_synchronize() {
        __kuser_memory_barrier();
    }
}
