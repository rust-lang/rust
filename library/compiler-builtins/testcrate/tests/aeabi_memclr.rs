#![cfg(all(
    target_arch = "arm",
    not(any(target_env = "gnu", target_env = "musl")),
    target_os = "linux",
    feature = "mem"
))]
#![feature(compiler_builtins_lib)]
#![feature(lang_items)]
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

extern "C" {
    fn __aeabi_memclr4(dest: *mut u8, n: usize);
    fn __aeabi_memset4(dest: *mut u8, n: usize, c: u32);
}

struct Aligned {
    array: [u8; 8],
    _alignment: [u32; 0],
}

impl Aligned {
    fn new() -> Self {
        Aligned {
            array: [0; 8],
            _alignment: [],
        }
    }
}

#[test]
fn memclr4() {
    let mut aligned = Aligned::new();
    assert_eq!(mem::align_of_val(&aligned), 4);
    let xs = &mut aligned.array;

    for n in 0..9 {
        unsafe {
            __aeabi_memset4(xs.as_mut_ptr(), n, 0xff);
            __aeabi_memclr4(xs.as_mut_ptr(), n);
        }

        assert!(xs[0..n].iter().all(|x| *x == 0));
    }
}
