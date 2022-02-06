use core::intrinsics;
use core::mem;

// Kernel-provided user-mode helper functions:
// https://www.kernel.org/doc/Documentation/arm/kernel_user_helpers.txt
unsafe fn __kuser_cmpxchg(oldval: u32, newval: u32, ptr: *mut u32) -> bool {
    let f: extern "C" fn(u32, u32, *mut u32) -> u32 = mem::transmute(0xffff0fc0u32);
    f(oldval, newval, ptr) == 0
}
unsafe fn __kuser_memory_barrier() {
    let f: extern "C" fn() = mem::transmute(0xffff0fa0u32);
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

// Generic atomic read-modify-write operation
unsafe fn atomic_rmw<T, F: Fn(u32) -> u32>(ptr: *mut T, f: F) -> u32 {
    let aligned_ptr = align_ptr(ptr);
    let (shift, mask) = get_shift_mask(ptr);

    loop {
        let curval_aligned = intrinsics::atomic_load_unordered(aligned_ptr);
        let curval = extract_aligned(curval_aligned, shift, mask);
        let newval = f(curval);
        let newval_aligned = insert_aligned(curval_aligned, newval, shift, mask);
        if __kuser_cmpxchg(curval_aligned, newval_aligned, aligned_ptr) {
            return curval;
        }
    }
}

// Generic atomic compare-exchange operation
unsafe fn atomic_cmpxchg<T>(ptr: *mut T, oldval: u32, newval: u32) -> u32 {
    let aligned_ptr = align_ptr(ptr);
    let (shift, mask) = get_shift_mask(ptr);

    loop {
        let curval_aligned = intrinsics::atomic_load_unordered(aligned_ptr);
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
    ($name:ident, $ty:ty, $op:expr) => {
        intrinsics! {
            pub unsafe extern "C" fn $name(ptr: *mut $ty, val: $ty) -> $ty {
                atomic_rmw(ptr, |x| $op(x as $ty, val) as u32) as $ty
            }
        }
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

atomic_rmw!(__sync_fetch_and_add_1, u8, |a: u8, b: u8| a.wrapping_add(b));
atomic_rmw!(__sync_fetch_and_add_2, u16, |a: u16, b: u16| a
    .wrapping_add(b));
atomic_rmw!(__sync_fetch_and_add_4, u32, |a: u32, b: u32| a
    .wrapping_add(b));

atomic_rmw!(__sync_fetch_and_sub_1, u8, |a: u8, b: u8| a.wrapping_sub(b));
atomic_rmw!(__sync_fetch_and_sub_2, u16, |a: u16, b: u16| a
    .wrapping_sub(b));
atomic_rmw!(__sync_fetch_and_sub_4, u32, |a: u32, b: u32| a
    .wrapping_sub(b));

atomic_rmw!(__sync_fetch_and_and_1, u8, |a: u8, b: u8| a & b);
atomic_rmw!(__sync_fetch_and_and_2, u16, |a: u16, b: u16| a & b);
atomic_rmw!(__sync_fetch_and_and_4, u32, |a: u32, b: u32| a & b);

atomic_rmw!(__sync_fetch_and_or_1, u8, |a: u8, b: u8| a | b);
atomic_rmw!(__sync_fetch_and_or_2, u16, |a: u16, b: u16| a | b);
atomic_rmw!(__sync_fetch_and_or_4, u32, |a: u32, b: u32| a | b);

atomic_rmw!(__sync_fetch_and_xor_1, u8, |a: u8, b: u8| a ^ b);
atomic_rmw!(__sync_fetch_and_xor_2, u16, |a: u16, b: u16| a ^ b);
atomic_rmw!(__sync_fetch_and_xor_4, u32, |a: u32, b: u32| a ^ b);

atomic_rmw!(__sync_fetch_and_nand_1, u8, |a: u8, b: u8| !(a & b));
atomic_rmw!(__sync_fetch_and_nand_2, u16, |a: u16, b: u16| !(a & b));
atomic_rmw!(__sync_fetch_and_nand_4, u32, |a: u32, b: u32| !(a & b));

atomic_rmw!(__sync_fetch_and_max_1, i8, |a: i8, b: i8| if a > b {
    a
} else {
    b
});
atomic_rmw!(__sync_fetch_and_max_2, i16, |a: i16, b: i16| if a > b {
    a
} else {
    b
});
atomic_rmw!(__sync_fetch_and_max_4, i32, |a: i32, b: i32| if a > b {
    a
} else {
    b
});

atomic_rmw!(__sync_fetch_and_umax_1, u8, |a: u8, b: u8| if a > b {
    a
} else {
    b
});
atomic_rmw!(__sync_fetch_and_umax_2, u16, |a: u16, b: u16| if a > b {
    a
} else {
    b
});
atomic_rmw!(__sync_fetch_and_umax_4, u32, |a: u32, b: u32| if a > b {
    a
} else {
    b
});

atomic_rmw!(__sync_fetch_and_min_1, i8, |a: i8, b: i8| if a < b {
    a
} else {
    b
});
atomic_rmw!(__sync_fetch_and_min_2, i16, |a: i16, b: i16| if a < b {
    a
} else {
    b
});
atomic_rmw!(__sync_fetch_and_min_4, i32, |a: i32, b: i32| if a < b {
    a
} else {
    b
});

atomic_rmw!(__sync_fetch_and_umin_1, u8, |a: u8, b: u8| if a < b {
    a
} else {
    b
});
atomic_rmw!(__sync_fetch_and_umin_2, u16, |a: u16, b: u16| if a < b {
    a
} else {
    b
});
atomic_rmw!(__sync_fetch_and_umin_4, u32, |a: u32, b: u32| if a < b {
    a
} else {
    b
});

atomic_rmw!(__sync_lock_test_and_set_1, u8, |_: u8, b: u8| b);
atomic_rmw!(__sync_lock_test_and_set_2, u16, |_: u16, b: u16| b);
atomic_rmw!(__sync_lock_test_and_set_4, u32, |_: u32, b: u32| b);

atomic_cmpxchg!(__sync_val_compare_and_swap_1, u8);
atomic_cmpxchg!(__sync_val_compare_and_swap_2, u16);
atomic_cmpxchg!(__sync_val_compare_and_swap_4, u32);

intrinsics! {
    pub unsafe extern "C" fn __sync_synchronize() {
        __kuser_memory_barrier();
    }
}
