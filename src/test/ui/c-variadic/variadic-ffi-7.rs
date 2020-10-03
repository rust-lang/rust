// run-pass
#![feature(c_variadic)]

macro_rules! variadic_tests {
    (generate) => {
        pub unsafe extern "C" fn sum(n: usize, mut ap: ...) -> usize {
            (0..n).fold(0usize, |acc, _| {
                acc + ap.arg::<usize>()
            })
        }

        pub unsafe extern "C" fn mixed_types(_: usize, mut ap: ...) {
            use std::ffi::{CString, CStr};

            assert_eq!(ap.arg::<f64>().floor(), 12.0f64.floor());
            assert_eq!(ap.arg::<u32>(), 0x10000001);
            assert_eq!(ap.arg::<i8>(), 0x42);
            let cstr = CStr::from_ptr(ap.arg::<*const i8>());
            assert!(cstr == &*CString::new("Hello").unwrap());
        }
    };

    (test_sum $p:path) => {
        unsafe {
            assert_eq!($p(5, 2, 4, 8, 16, 32), 62);
            assert_eq!($p(4, 10, 8, 8, 16, 2), 42);
        }
    };

    (test_mixed_types $p:path) => {
        unsafe {
            use std::ffi::CString;

            $p(0, 12.56f64, 0x10000001, 0x42, CString::new("Hello").unwrap().as_ptr());
        }
    }
}

mod foo {
    variadic_tests!(generate);
}

struct Foo;

impl Foo {
    variadic_tests!(generate);
}

fn main() {
    variadic_tests!(test_sum foo::sum);
    variadic_tests!(test_sum Foo::sum);

    variadic_tests!(test_mixed_types foo::mixed_types);
    variadic_tests!(test_mixed_types Foo::mixed_types);
}
