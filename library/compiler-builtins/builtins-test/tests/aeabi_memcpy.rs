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

macro_rules! panic {
    ($($tt:tt)*) => {
        upanic!($($tt)*);
    };
}

// SAFETY: defined in  compiler-builtins
unsafe extern "aapcs" {
    fn __aeabi_memcpy(dest: *mut u8, src: *const u8, n: usize);
    fn __aeabi_memcpy4(dest: *mut u8, src: *const u8, n: usize);
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
fn memcpy() {
    let mut dest = [0; 4];
    let src = [0xde, 0xad, 0xbe, 0xef];

    for n in 0..dest.len() {
        dest.copy_from_slice(&[0; 4]);

        unsafe { __aeabi_memcpy(dest.as_mut_ptr(), src.as_ptr(), n) }

        assert_eq!(&dest[0..n], &src[0..n])
    }
}

#[test]
fn memcpy4() {
    let mut aligned = Aligned::new([0; 8]);
    let dest = &mut aligned.array;
    let src = [0xde, 0xad, 0xbe, 0xef, 0xba, 0xad, 0xf0, 0x0d];

    for n in 0..dest.len() {
        dest.copy_from_slice(&[0; 8]);

        unsafe { __aeabi_memcpy4(dest.as_mut_ptr(), src.as_ptr(), n) }

        assert_eq!(&dest[0..n], &src[0..n])
    }
}
