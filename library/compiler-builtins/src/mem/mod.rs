// Trying to satisfy clippy here is hopeless
#![allow(clippy::style)]

#[allow(warnings)]
#[cfg(target_pointer_width = "16")]
type c_int = i16;
#[allow(warnings)]
#[cfg(not(target_pointer_width = "16"))]
type c_int = i32;

use core::intrinsics::{atomic_load_unordered, atomic_store_unordered, exact_div};
use core::mem;
use core::ops::{BitOr, Shl};

// memcpy/memmove/memset have optimized implementations on some architectures
#[cfg_attr(
    all(not(feature = "no-asm"), target_arch = "x86_64"),
    path = "x86_64.rs"
)]
mod impls;

#[cfg_attr(all(feature = "mem", not(feature = "mangled-names")), no_mangle)]
#[cfg_attr(not(all(target_os = "windows", target_env = "gnu")), linkage = "weak")]
pub unsafe extern "C" fn memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8 {
    impls::copy_forward(dest, src, n);
    dest
}

#[cfg_attr(all(feature = "mem", not(feature = "mangled-names")), no_mangle)]
#[cfg_attr(not(all(target_os = "windows", target_env = "gnu")), linkage = "weak")]
pub unsafe extern "C" fn memmove(dest: *mut u8, src: *const u8, n: usize) -> *mut u8 {
    let delta = (dest as usize).wrapping_sub(src as usize);
    if delta >= n {
        // We can copy forwards because either dest is far enough ahead of src,
        // or src is ahead of dest (and delta overflowed).
        impls::copy_forward(dest, src, n);
    } else {
        impls::copy_backward(dest, src, n);
    }
    dest
}

#[cfg_attr(all(feature = "mem", not(feature = "mangled-names")), no_mangle)]
#[cfg_attr(not(all(target_os = "windows", target_env = "gnu")), linkage = "weak")]
pub unsafe extern "C" fn memset(s: *mut u8, c: c_int, n: usize) -> *mut u8 {
    impls::set_bytes(s, c as u8, n);
    s
}

#[cfg_attr(all(feature = "mem", not(feature = "mangled-names")), no_mangle)]
#[cfg_attr(not(all(target_os = "windows", target_env = "gnu")), linkage = "weak")]
pub unsafe extern "C" fn memcmp(s1: *const u8, s2: *const u8, n: usize) -> i32 {
    let mut i = 0;
    while i < n {
        let a = *s1.add(i);
        let b = *s2.add(i);
        if a != b {
            return a as i32 - b as i32;
        }
        i += 1;
    }
    0
}

#[cfg_attr(all(feature = "mem", not(feature = "mangled-names")), no_mangle)]
#[cfg_attr(not(all(target_os = "windows", target_env = "gnu")), linkage = "weak")]
pub unsafe extern "C" fn bcmp(s1: *const u8, s2: *const u8, n: usize) -> i32 {
    memcmp(s1, s2, n)
}

// `bytes` must be a multiple of `mem::size_of::<T>()`
fn memcpy_element_unordered_atomic<T: Copy>(dest: *mut T, src: *const T, bytes: usize) {
    unsafe {
        let n = exact_div(bytes, mem::size_of::<T>());
        let mut i = 0;
        while i < n {
            atomic_store_unordered(dest.add(i), atomic_load_unordered(src.add(i)));
            i += 1;
        }
    }
}

// `bytes` must be a multiple of `mem::size_of::<T>()`
fn memmove_element_unordered_atomic<T: Copy>(dest: *mut T, src: *const T, bytes: usize) {
    unsafe {
        let n = exact_div(bytes, mem::size_of::<T>());
        if src < dest as *const T {
            // copy from end
            let mut i = n;
            while i != 0 {
                i -= 1;
                atomic_store_unordered(dest.add(i), atomic_load_unordered(src.add(i)));
            }
        } else {
            // copy from beginning
            let mut i = 0;
            while i < n {
                atomic_store_unordered(dest.add(i), atomic_load_unordered(src.add(i)));
                i += 1;
            }
        }
    }
}

// `T` must be a primitive integer type, and `bytes` must be a multiple of `mem::size_of::<T>()`
fn memset_element_unordered_atomic<T>(s: *mut T, c: u8, bytes: usize)
where
    T: Copy + From<u8> + Shl<u32, Output = T> + BitOr<T, Output = T>,
{
    unsafe {
        let n = exact_div(bytes, mem::size_of::<T>());

        // Construct a value of type `T` consisting of repeated `c`
        // bytes, to let us ensure we write each `T` atomically.
        let mut x = T::from(c);
        let mut i = 1;
        while i < mem::size_of::<T>() {
            x = x << 8 | T::from(c);
            i += 1;
        }

        // Write it to `s`
        let mut i = 0;
        while i < n {
            atomic_store_unordered(s.add(i), x);
            i += 1;
        }
    }
}

intrinsics! {
    #[cfg(target_has_atomic_load_store = "8")]
    pub extern "C" fn __llvm_memcpy_element_unordered_atomic_1(dest: *mut u8, src: *const u8, bytes: usize) -> () {
        memcpy_element_unordered_atomic(dest, src, bytes);
    }
    #[cfg(target_has_atomic_load_store = "16")]
    pub extern "C" fn __llvm_memcpy_element_unordered_atomic_2(dest: *mut u16, src: *const u16, bytes: usize) -> () {
        memcpy_element_unordered_atomic(dest, src, bytes);
    }
    #[cfg(target_has_atomic_load_store = "32")]
    pub extern "C" fn __llvm_memcpy_element_unordered_atomic_4(dest: *mut u32, src: *const u32, bytes: usize) -> () {
        memcpy_element_unordered_atomic(dest, src, bytes);
    }
    #[cfg(target_has_atomic_load_store = "64")]
    pub extern "C" fn __llvm_memcpy_element_unordered_atomic_8(dest: *mut u64, src: *const u64, bytes: usize) -> () {
        memcpy_element_unordered_atomic(dest, src, bytes);
    }
    #[cfg(target_has_atomic_load_store = "128")]
    pub extern "C" fn __llvm_memcpy_element_unordered_atomic_16(dest: *mut u128, src: *const u128, bytes: usize) -> () {
        memcpy_element_unordered_atomic(dest, src, bytes);
    }

    #[cfg(target_has_atomic_load_store = "8")]
    pub extern "C" fn __llvm_memmove_element_unordered_atomic_1(dest: *mut u8, src: *const u8, bytes: usize) -> () {
        memmove_element_unordered_atomic(dest, src, bytes);
    }
    #[cfg(target_has_atomic_load_store = "16")]
    pub extern "C" fn __llvm_memmove_element_unordered_atomic_2(dest: *mut u16, src: *const u16, bytes: usize) -> () {
        memmove_element_unordered_atomic(dest, src, bytes);
    }
    #[cfg(target_has_atomic_load_store = "32")]
    pub extern "C" fn __llvm_memmove_element_unordered_atomic_4(dest: *mut u32, src: *const u32, bytes: usize) -> () {
        memmove_element_unordered_atomic(dest, src, bytes);
    }
    #[cfg(target_has_atomic_load_store = "64")]
    pub extern "C" fn __llvm_memmove_element_unordered_atomic_8(dest: *mut u64, src: *const u64, bytes: usize) -> () {
        memmove_element_unordered_atomic(dest, src, bytes);
    }
    #[cfg(target_has_atomic_load_store = "128")]
    pub extern "C" fn __llvm_memmove_element_unordered_atomic_16(dest: *mut u128, src: *const u128, bytes: usize) -> () {
        memmove_element_unordered_atomic(dest, src, bytes);
    }

    #[cfg(target_has_atomic_load_store = "8")]
    pub extern "C" fn __llvm_memset_element_unordered_atomic_1(s: *mut u8, c: u8, bytes: usize) -> () {
        memset_element_unordered_atomic(s, c, bytes);
    }
    #[cfg(target_has_atomic_load_store = "16")]
    pub extern "C" fn __llvm_memset_element_unordered_atomic_2(s: *mut u16, c: u8, bytes: usize) -> () {
        memset_element_unordered_atomic(s, c, bytes);
    }
    #[cfg(target_has_atomic_load_store = "32")]
    pub extern "C" fn __llvm_memset_element_unordered_atomic_4(s: *mut u32, c: u8, bytes: usize) -> () {
        memset_element_unordered_atomic(s, c, bytes);
    }
    #[cfg(target_has_atomic_load_store = "64")]
    pub extern "C" fn __llvm_memset_element_unordered_atomic_8(s: *mut u64, c: u8, bytes: usize) -> () {
        memset_element_unordered_atomic(s, c, bytes);
    }
    #[cfg(target_has_atomic_load_store = "128")]
    pub extern "C" fn __llvm_memset_element_unordered_atomic_16(s: *mut u128, c: u8, bytes: usize) -> () {
        memset_element_unordered_atomic(s, c, bytes);
    }
}
