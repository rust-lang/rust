#![feature(core_intrinsics, const_mut_refs, const_swap)]
#![crate_type = "rlib"]

//! This module tests if `swap_nonoverlapping_single` works properly in const contexts.

use std::intrinsics::swap_nonoverlapping_single;

pub const OK_A: () = {
    let mut a = 0i32;
    let mut b = 5i32;
    unsafe {
        swap_nonoverlapping_single(&mut a, &mut b);
    }
    assert!(a == 5, "Must NOT fail.");
    assert!(b == 0, "Must NOT fail.");
};

pub const ERR_A0: () = {
    let mut a = 0i32;
    let mut b = 5i32;
    unsafe {
        swap_nonoverlapping_single(&mut a, &mut b);
    }

    assert!(a != 5, "Must fail."); //~ ERROR evaluation of constant value failed
};

pub const ERR_A1: () = {
    let mut a = 0i32;
    let mut b = 5i32;
    unsafe {
        swap_nonoverlapping_single(&mut a, &mut b);
    }

    assert!(b != 0, "Must fail."); //~ ERROR evaluation of constant value failed
};

// This must NOT fail.
pub const B: () = {
    let mut a = 0i32;
    let mut b = 5i32;
    unsafe {
        swap_nonoverlapping_single(&mut a, &mut b);
    }
    unsafe {
        swap_nonoverlapping_single(&mut a, &mut b);
    }
    assert!(a == 0, "Must NOT fail.");
    assert!(b == 5, "Must NOT fail.");
};

pub const ERR_B0: () = {
    let mut a = 0i32;
    let mut b = 5i32;
    unsafe {
        swap_nonoverlapping_single(&mut a, &mut b);
    }
    unsafe {
        swap_nonoverlapping_single(&mut a, &mut b);
    }

    assert!(a != 0, "Must fail."); //~ ERROR evaluation of constant value failed
};

pub const ERR_B1: () = {
    let mut a = 0i32;
    let mut b = 5i32;
    unsafe {
        swap_nonoverlapping_single(&mut a, &mut b);
    }
    unsafe {
        swap_nonoverlapping_single(&mut a, &mut b);
    }

    assert!(b != 5, "Must fail."); //~ ERROR evaluation of constant value failed
};

// This must NOT fail.
pub const NON_OVERLAPPING_PTRS: () = {
    let mut chunk = [0_i32, 1, 2, 3];

    let ptr = chunk.as_mut_ptr();
    let ptr2 = unsafe { ptr.add(2) };
    let x: &mut [i32; 2] = unsafe { &mut *ptr.cast() };
    let y: &mut [i32; 2] = unsafe { &mut *ptr2.cast() };
    unsafe {
        swap_nonoverlapping_single(x, y);
    }

    assert!(matches!(chunk, [2, 3, 0, 1]), "Must NOT fail.");
};

pub const OVERLAPPING_PTRS_0: () = {
    let mut chunk = [0_i32, 1, 2, 3];

    let ptr = chunk.as_mut_ptr();
    let ptr2 = unsafe { ptr.add(1) };
    let x: &mut [i32; 2] = unsafe { &mut *ptr.cast() };
    let y: &mut [i32; 2] = unsafe { &mut *ptr2.cast() };

    unsafe {
        swap_nonoverlapping_single(x, y); //~ ERROR evaluation of constant value failed
    }
};

pub const OVERLAPPING_PTRS_1: () = {
    let mut val = 7;

    let ptr: *mut _ = &mut val;
    let x: &mut i32 = unsafe { &mut *ptr };
    let y: &mut i32 = unsafe { &mut *ptr };

    unsafe {
        swap_nonoverlapping_single(x, y); //~ ERROR evaluation of constant value failed
    }
};

pub const OK_STRUCT: () = {
    struct Adt {
        fl: bool,
        val: usize,
    }
    let mut a = Adt { fl: false, val: 10 };
    let mut b = Adt { fl: true, val: 77 };

    unsafe {
        swap_nonoverlapping_single(&mut a, &mut b);
    }

    assert!(matches!(a, Adt { fl: true, val: 77 }), "Must NOT fail.");
    assert!(matches!(b, Adt { fl: false, val: 10 }), "Must NOT fail.");
};
