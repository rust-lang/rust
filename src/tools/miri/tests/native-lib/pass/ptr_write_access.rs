// Only works on Unix targets
//@ignore-target: windows wasm
//@only-on-host
//@compile-flags: -Zmiri-permissive-provenance

#![feature(box_as_ptr)]

use std::mem::MaybeUninit;
use std::ptr::null;

fn main() {
    test_increment_int();
    test_init_int();
    test_init_array();
    test_init_static_inner();
    test_exposed();
    test_swap_ptr();
    test_swap_ptr_tuple();
    test_overwrite_dangling();
    test_pass_dangling();
    test_swap_ptr_triple_dangling();
    test_return_ptr();
}

/// Test function that modifies an int.
fn test_increment_int() {
    extern "C" {
        fn increment_int(ptr: *mut i32);
    }

    let mut x = 11;

    unsafe { increment_int(&mut x) };
    assert_eq!(x, 12);
}

/// Test function that initializes an int.
fn test_init_int() {
    extern "C" {
        fn init_int(ptr: *mut i32, val: i32);
    }

    let mut x = MaybeUninit::<i32>::uninit();
    let val = 21;

    let x = unsafe {
        init_int(x.as_mut_ptr(), val);
        x.assume_init()
    };
    assert_eq!(x, val);
}

/// Test function that initializes an array.
fn test_init_array() {
    extern "C" {
        fn init_array(ptr: *mut i32, len: usize, val: i32);
    }

    const LEN: usize = 3;
    let mut array = MaybeUninit::<[i32; LEN]>::uninit();
    let val = 31;

    let array = unsafe {
        init_array(array.as_mut_ptr().cast::<i32>(), LEN, val);
        array.assume_init()
    };
    assert_eq!(array, [val; LEN]);
}

/// Test function that initializes an int pointed to by an immutable static.
fn test_init_static_inner() {
    #[repr(C)]
    struct SyncPtr {
        ptr: *mut i32,
    }
    unsafe impl Sync for SyncPtr {}

    extern "C" {
        fn init_static_inner(s_ptr: *const SyncPtr, val: i32);
    }

    static mut INNER: MaybeUninit<i32> = MaybeUninit::uninit();
    #[allow(static_mut_refs)]
    static STATIC: SyncPtr = SyncPtr { ptr: unsafe { INNER.as_mut_ptr() } };
    let val = 41;

    let inner = unsafe {
        init_static_inner(&STATIC, val);
        INNER.assume_init()
    };
    assert_eq!(inner, val);
}

// Test function that marks an allocation as exposed.
fn test_exposed() {
    extern "C" {
        fn ignore_ptr(ptr: *const i32);
    }

    let x = 51;
    let ptr = &raw const x;
    let p = ptr.addr();

    unsafe { ignore_ptr(ptr) };
    assert_eq!(unsafe { *(p as *const i32) }, x);
}

/// Test function that swaps two pointers and exposes the alloc of an int.
fn test_swap_ptr() {
    extern "C" {
        fn swap_ptr(pptr0: *mut *const i32, pptr1: *mut *const i32);
    }

    let x = 61;
    let (mut ptr0, mut ptr1) = (&raw const x, null());

    unsafe { swap_ptr(&mut ptr0, &mut ptr1) };
    assert_eq!(unsafe { *ptr1 }, x);
}

/// Test function that swaps two pointers in a struct and exposes the alloc of an int.
fn test_swap_ptr_tuple() {
    #[repr(C)]
    struct Tuple {
        ptr0: *const i32,
        ptr1: *const i32,
    }

    extern "C" {
        fn swap_ptr_tuple(t_ptr: *mut Tuple);
    }

    let x = 71;
    let mut tuple = Tuple { ptr0: &raw const x, ptr1: null() };

    unsafe { swap_ptr_tuple(&mut tuple) }
    assert_eq!(unsafe { *tuple.ptr1 }, x);
}

/// Test function that interacts with a dangling pointer.
fn test_overwrite_dangling() {
    extern "C" {
        fn overwrite_ptr(pptr: *mut *const i32);
    }

    let b = Box::new(81);
    let mut ptr = Box::as_ptr(&b);
    drop(b);

    unsafe { overwrite_ptr(&mut ptr) };
    assert_eq!(ptr, null());
}

/// Test function that passes a dangling pointer.
fn test_pass_dangling() {
    extern "C" {
        fn ignore_ptr(ptr: *const i32);
    }

    let b = Box::new(91);
    let ptr = Box::as_ptr(&b);
    drop(b);

    unsafe { ignore_ptr(ptr) };
}

/// Test function that interacts with a struct storing a dangling pointer.
fn test_swap_ptr_triple_dangling() {
    #[repr(C)]
    struct Triple {
        ptr0: *const i32,
        ptr1: *const i32,
        ptr2: *const i32,
    }

    extern "C" {
        fn swap_ptr_triple_dangling(t_ptr: *const Triple);
    }

    let x = 101;
    let b = Box::new(111);
    let ptr = Box::as_ptr(&b);
    drop(b);
    let z = 121;
    let triple = Triple { ptr0: &raw const x, ptr1: ptr, ptr2: &raw const z };

    unsafe { swap_ptr_triple_dangling(&triple) }
    assert_eq!(unsafe { *triple.ptr2 }, x);
}

/// Test function that directly returns its pointer argument.
fn test_return_ptr() {
    extern "C" {
        fn return_ptr(ptr: *const i32) -> *const i32;
    }

    let x = 131;
    let ptr = &raw const x;

    let ptr = unsafe { return_ptr(ptr) };
    assert_eq!(unsafe { *ptr }, x);
}
