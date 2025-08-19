#![cfg(all(
    target_arch = "arm",
    not(any(target_env = "gnu", target_env = "musl")),
    target_os = "linux",
    feature = "mem"
))]
#![feature(compiler_builtins_lib)]
#![no_std]

extern crate compiler_builtins;

// test runner
extern crate utest_cortex_m_qemu;

// overrides `panic!`
#[macro_use]
extern crate utest_macros;

use core::mem;

macro_rules! panic {
    ($($tt:tt)*) => {
        upanic!($($tt)*);
    };
}

// SAFETY: defined in  compiler-builtins
unsafe extern "aapcs" {
    fn __aeabi_memset4(dest: *mut u8, n: usize, c: u32);
}

struct Aligned {
    array: [u8; 8],
    _alignment: [u32; 0],
}

impl Aligned {
    fn new(array: [u8; 8]) -> Self {
        Aligned {
            array: array,
            _alignment: [],
        }
    }
}

#[test]
fn zero() {
    let mut aligned = Aligned::new([0u8; 8]);
    assert_eq!(mem::align_of_val(&aligned), 4);
    let xs = &mut aligned.array;
    let c = 0xdeadbeef;

    unsafe { __aeabi_memset4(xs.as_mut_ptr(), 0, c) }

    assert_eq!(*xs, [0; 8]);

    let mut aligned = Aligned::new([1u8; 8]);
    assert_eq!(mem::align_of_val(&aligned), 4);
    let xs = &mut aligned.array;
    let c = 0xdeadbeef;

    unsafe { __aeabi_memset4(xs.as_mut_ptr(), 0, c) }

    assert_eq!(*xs, [1; 8]);
}

#[test]
fn one() {
    let mut aligned = Aligned::new([0u8; 8]);
    assert_eq!(mem::align_of_val(&aligned), 4);
    let xs = &mut aligned.array;
    let n = 1;
    let c = 0xdeadbeef;

    unsafe { __aeabi_memset4(xs.as_mut_ptr(), n, c) }

    assert_eq!(*xs, [0xef, 0, 0, 0, 0, 0, 0, 0]);

    let mut aligned = Aligned::new([1u8; 8]);
    assert_eq!(mem::align_of_val(&aligned), 4);
    let xs = &mut aligned.array;
    let c = 0xdeadbeef;

    unsafe { __aeabi_memset4(xs.as_mut_ptr(), n, c) }

    assert_eq!(*xs, [0xef, 1, 1, 1, 1, 1, 1, 1]);
}

#[test]
fn two() {
    let mut aligned = Aligned::new([0u8; 8]);
    assert_eq!(mem::align_of_val(&aligned), 4);
    let xs = &mut aligned.array;
    let n = 2;
    let c = 0xdeadbeef;

    unsafe { __aeabi_memset4(xs.as_mut_ptr(), n, c) }

    assert_eq!(*xs, [0xef, 0xef, 0, 0, 0, 0, 0, 0]);

    let mut aligned = Aligned::new([1u8; 8]);
    assert_eq!(mem::align_of_val(&aligned), 4);
    let xs = &mut aligned.array;
    let c = 0xdeadbeef;

    unsafe { __aeabi_memset4(xs.as_mut_ptr(), n, c) }

    assert_eq!(*xs, [0xef, 0xef, 1, 1, 1, 1, 1, 1]);
}

#[test]
fn three() {
    let mut aligned = Aligned::new([0u8; 8]);
    assert_eq!(mem::align_of_val(&aligned), 4);
    let xs = &mut aligned.array;
    let n = 3;
    let c = 0xdeadbeef;

    unsafe { __aeabi_memset4(xs.as_mut_ptr(), n, c) }

    assert_eq!(*xs, [0xef, 0xef, 0xef, 0, 0, 0, 0, 0]);

    let mut aligned = Aligned::new([1u8; 8]);
    assert_eq!(mem::align_of_val(&aligned), 4);
    let xs = &mut aligned.array;
    let c = 0xdeadbeef;

    unsafe { __aeabi_memset4(xs.as_mut_ptr(), n, c) }

    assert_eq!(*xs, [0xef, 0xef, 0xef, 1, 1, 1, 1, 1]);
}

#[test]
fn four() {
    let mut aligned = Aligned::new([0u8; 8]);
    assert_eq!(mem::align_of_val(&aligned), 4);
    let xs = &mut aligned.array;
    let n = 4;
    let c = 0xdeadbeef;

    unsafe { __aeabi_memset4(xs.as_mut_ptr(), n, c) }

    assert_eq!(*xs, [0xef, 0xef, 0xef, 0xef, 0, 0, 0, 0]);

    let mut aligned = Aligned::new([1u8; 8]);
    assert_eq!(mem::align_of_val(&aligned), 4);
    let xs = &mut aligned.array;
    let c = 0xdeadbeef;

    unsafe { __aeabi_memset4(xs.as_mut_ptr(), n, c) }

    assert_eq!(*xs, [0xef, 0xef, 0xef, 0xef, 1, 1, 1, 1]);
}

#[test]
fn five() {
    let mut aligned = Aligned::new([0u8; 8]);
    assert_eq!(mem::align_of_val(&aligned), 4);
    let xs = &mut aligned.array;
    let n = 5;
    let c = 0xdeadbeef;

    unsafe { __aeabi_memset4(xs.as_mut_ptr(), n, c) }

    assert_eq!(*xs, [0xef, 0xef, 0xef, 0xef, 0xef, 0, 0, 0]);

    let mut aligned = Aligned::new([1u8; 8]);
    assert_eq!(mem::align_of_val(&aligned), 4);
    let xs = &mut aligned.array;
    let c = 0xdeadbeef;

    unsafe { __aeabi_memset4(xs.as_mut_ptr(), n, c) }

    assert_eq!(*xs, [0xef, 0xef, 0xef, 0xef, 0xef, 1, 1, 1]);
}

#[test]
fn six() {
    let mut aligned = Aligned::new([0u8; 8]);
    assert_eq!(mem::align_of_val(&aligned), 4);
    let xs = &mut aligned.array;
    let n = 6;
    let c = 0xdeadbeef;

    unsafe { __aeabi_memset4(xs.as_mut_ptr(), n, c) }

    assert_eq!(*xs, [0xef, 0xef, 0xef, 0xef, 0xef, 0xef, 0, 0]);

    let mut aligned = Aligned::new([1u8; 8]);
    assert_eq!(mem::align_of_val(&aligned), 4);
    let xs = &mut aligned.array;
    let c = 0xdeadbeef;

    unsafe { __aeabi_memset4(xs.as_mut_ptr(), n, c) }

    assert_eq!(*xs, [0xef, 0xef, 0xef, 0xef, 0xef, 0xef, 1, 1]);
}

#[test]
fn seven() {
    let mut aligned = Aligned::new([0u8; 8]);
    assert_eq!(mem::align_of_val(&aligned), 4);
    let xs = &mut aligned.array;
    let n = 7;
    let c = 0xdeadbeef;

    unsafe { __aeabi_memset4(xs.as_mut_ptr(), n, c) }

    assert_eq!(*xs, [0xef, 0xef, 0xef, 0xef, 0xef, 0xef, 0xef, 0]);

    let mut aligned = Aligned::new([1u8; 8]);
    assert_eq!(mem::align_of_val(&aligned), 4);
    let xs = &mut aligned.array;
    let c = 0xdeadbeef;

    unsafe { __aeabi_memset4(xs.as_mut_ptr(), n, c) }

    assert_eq!(*xs, [0xef, 0xef, 0xef, 0xef, 0xef, 0xef, 0xef, 1]);
}

#[test]
fn eight() {
    let mut aligned = Aligned::new([0u8; 8]);
    assert_eq!(mem::align_of_val(&aligned), 4);
    let xs = &mut aligned.array;
    let n = 8;
    let c = 0xdeadbeef;

    unsafe { __aeabi_memset4(xs.as_mut_ptr(), n, c) }

    assert_eq!(*xs, [0xef, 0xef, 0xef, 0xef, 0xef, 0xef, 0xef, 0xef]);

    let mut aligned = Aligned::new([1u8; 8]);
    assert_eq!(mem::align_of_val(&aligned), 4);
    let xs = &mut aligned.array;
    let c = 0xdeadbeef;

    unsafe { __aeabi_memset4(xs.as_mut_ptr(), n, c) }

    assert_eq!(*xs, [0xef, 0xef, 0xef, 0xef, 0xef, 0xef, 0xef, 0xef]);
}
