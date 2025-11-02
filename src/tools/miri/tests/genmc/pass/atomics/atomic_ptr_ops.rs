//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows

// Test several operations on atomic pointers.

#![no_main]

#[path = "../../../utils/mod.rs"]
mod utils;

use std::fmt::{Debug, Write};
use std::sync::atomic::AtomicPtr;
use std::sync::atomic::Ordering::*;

use crate::utils::*;

static mut X: u64 = 0;
static mut Y: u64 = 0;

fn assert_equals<T: Eq + Copy + Debug>(a: T, b: T) {
    if a != b {
        writeln!(MiriStderr, "{:?}, {:?}", a, b).ok();
        std::process::abort();
    }
}

/// Check that two pointers are equal and stores to one update the value read from the other.
unsafe fn pointers_equal(a: *mut u64, b: *mut u64) {
    assert_equals(a, b);
    assert_equals(*a, *b);
    *a = 42;
    assert_equals(*a, 42);
    assert_equals(*b, 42);
    *b = 0xAA;
    assert_equals(*a, 0xAA);
    assert_equals(*b, 0xAA);
}

unsafe fn test_load_store_exchange() {
    let atomic_ptr: AtomicPtr<u64> = AtomicPtr::new(&raw mut X);

    // Atomic load can read the initial value.
    pointers_equal(atomic_ptr.load(SeqCst), &raw mut X);
    // Atomic store works as expected.
    atomic_ptr.store(&raw mut Y, SeqCst);
    pointers_equal(atomic_ptr.load(SeqCst), &raw mut Y);
    // We can read the value of the atomic store non-atomically.
    pointers_equal(*atomic_ptr.as_ptr(), &raw mut Y);
    // We can read the value of a non-atomic store atomically.
    *atomic_ptr.as_ptr() = &raw mut X;
    pointers_equal(atomic_ptr.load(SeqCst), &raw mut X);

    // Atomic swap must return the old value and store the new one.
    *atomic_ptr.as_ptr() = &raw mut Y; // Test that we can read this non-atomic store using `swap`.
    pointers_equal(atomic_ptr.swap(&raw mut X, SeqCst), &raw mut Y);
    pointers_equal(*atomic_ptr.as_ptr(), &raw mut X);
    pointers_equal(atomic_ptr.load(SeqCst), &raw mut X);

    // Failing compare_exchange (wrong expected pointer).
    match atomic_ptr.compare_exchange(&raw mut Y, std::ptr::null_mut(), SeqCst, SeqCst) {
        Ok(_ptr) => std::process::abort(),
        Err(ptr) => pointers_equal(ptr, &raw mut X),
    }
    // Non-atomic read value should also be unchanged by a failing compare_exchange.
    pointers_equal(*atomic_ptr.as_ptr(), &raw mut X);

    // Failing compare_exchange (null).
    match atomic_ptr.compare_exchange(std::ptr::null_mut(), std::ptr::null_mut(), SeqCst, SeqCst) {
        Ok(_ptr) => std::process::abort(),
        Err(ptr) => pointers_equal(ptr, &raw mut X),
    }
    // Non-atomic read value should also be unchanged by a failing compare_exchange.
    pointers_equal(*atomic_ptr.as_ptr(), &raw mut X);

    // Successful compare_exchange.
    match atomic_ptr.compare_exchange(&raw mut X, &raw mut Y, SeqCst, SeqCst) {
        Ok(ptr) => pointers_equal(ptr, &raw mut X),
        Err(_ptr) => std::process::abort(),
    }
    // compare_exchange should update the pointer.
    pointers_equal(atomic_ptr.load(SeqCst), &raw mut Y);
    pointers_equal(*atomic_ptr.as_ptr(), &raw mut Y);
}

unsafe fn test_add_sub() {
    const LEN: usize = 16;
    let mut array: [u64; LEN] = std::array::from_fn(|i| i as u64);
    let atomic_ptr: AtomicPtr<u64> = AtomicPtr::new(&raw mut array[0]);

    // Each element of the array should be reachable using `fetch_ptr_add`.
    // All pointers must stay valid.
    for i in 0..LEN {
        let ptr = atomic_ptr.fetch_ptr_add(1, SeqCst);
        pointers_equal(ptr, &raw mut array[i]);
    }
    // This should return the pointer back to the start of the array.
    let ptr = atomic_ptr.fetch_ptr_sub(LEN, SeqCst);
    pointers_equal(ptr.offset(-(LEN as isize)), &raw mut array[0]);
    pointers_equal(atomic_ptr.load(SeqCst), &raw mut array[0]);

    let array_mid_ptr = &raw mut array[LEN / 2];
    for i in 0..size_of::<u64>() {
        atomic_ptr.store(array_mid_ptr, SeqCst); // Reset to test `byte_add` and `byte_sub`.
        pointers_equal(array_mid_ptr, atomic_ptr.fetch_byte_add(i, SeqCst));
        if array_mid_ptr.byte_add(i) != atomic_ptr.load(SeqCst) {
            std::process::abort();
        }
        if array_mid_ptr.byte_add(i) != atomic_ptr.fetch_byte_sub(i, SeqCst) {
            std::process::abort();
        }
        pointers_equal(array_mid_ptr, atomic_ptr.load(SeqCst));
    }
}

unsafe fn test_and_or_xor() {
    const LEN: usize = 16;
    #[repr(align(1024))] // Aligned to size 16 * 8 bytes.
    struct AlignedArray([u64; LEN]);

    let mut array = AlignedArray(std::array::from_fn(|i| i as u64 * 10));
    let array_ptr = &raw mut array.0[0];
    let atomic_ptr: AtomicPtr<u64> = AtomicPtr::new(array_ptr);

    // Test no-op arguments.
    assert_equals(array_ptr, atomic_ptr.fetch_or(0, SeqCst));
    assert_equals(array_ptr, atomic_ptr.fetch_xor(0, SeqCst));
    assert_equals(array_ptr, atomic_ptr.fetch_and(!0, SeqCst));
    assert_equals(array_ptr, atomic_ptr.load(SeqCst));

    // Test identity arguments.
    let array_addr = array_ptr as usize;
    assert_equals(array_ptr, atomic_ptr.fetch_or(array_addr, SeqCst));
    assert_equals(array_ptr, atomic_ptr.load(SeqCst));
    assert_equals(array_ptr, atomic_ptr.fetch_and(array_addr, SeqCst));
    assert_equals(array_ptr, atomic_ptr.load(SeqCst));
    assert_equals(array_ptr, atomic_ptr.fetch_xor(array_addr, SeqCst));
    assert_equals(std::ptr::null_mut(), atomic_ptr.load(SeqCst)); // `null_mut` is guaranteed to have address 0.

    // Test moving within an allocation.
    // The array is aligned to 64 bytes, so we can change which element we point by or/and/xor-ing the address.
    let index = LEN / 2; // Choose an index in the middle of the array.
    let offset = index * size_of::<u64>();
    let array_mid_ptr = &raw mut array.0[index];

    atomic_ptr.store(array_ptr, SeqCst); // Reset to test `or`.
    assert_equals(array_ptr, atomic_ptr.fetch_or(offset, SeqCst));
    assert_equals(array_mid_ptr, atomic_ptr.load(SeqCst));

    atomic_ptr.store(array_ptr, SeqCst); // Reset to test `xor`.
    assert_equals(array_ptr, atomic_ptr.fetch_xor(offset, SeqCst));
    assert_equals(array_mid_ptr, atomic_ptr.load(SeqCst));
    assert_equals(array_mid_ptr, atomic_ptr.fetch_xor(offset, SeqCst)); // two xor should yield original value.
    assert_equals(array_ptr, atomic_ptr.load(SeqCst));

    let mask = !(u64::BITS as usize - 1);
    for i in 0..size_of::<u64>() {
        // We offset the pointer by `i` bytes, making it unaligned.
        let offset_ptr = array_ptr.byte_add(i);
        atomic_ptr.store(array_ptr, SeqCst);
        // `fetch_byte_add` should return the old value.
        assert_equals(array_ptr, atomic_ptr.fetch_byte_add(i, SeqCst));
        // `ptr::byte_add` and `AtomicPtr::fetch_byte_add` should give the same result.
        if offset_ptr != atomic_ptr.fetch_and(mask, SeqCst) {
            std::process::abort();
        }
        // Masking off the last bits should restore the pointer.
        assert_equals(array_ptr, atomic_ptr.load(SeqCst));
    }
}

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        test_load_store_exchange();
        test_add_sub();
        test_and_or_xor();
        0
    }
}
