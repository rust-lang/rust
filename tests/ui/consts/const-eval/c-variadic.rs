//@ edition: 2021
//@ run-pass
//@ ignore-backends: gcc

#![feature(c_variadic)]
#![feature(const_c_variadic)]
#![feature(const_destruct)]
#![feature(c_variadic_const)]
#![feature(const_cmp)]
#![feature(const_trait_impl)]

use std::ffi::*;

fn ignores_arguments() {
    const unsafe extern "C" fn variadic(_: ...) {}

    const {
        unsafe { variadic() };
        unsafe { variadic(1, 2, 3) };
    }
}

fn echo() {
    const unsafe extern "C" fn variadic(mut ap: ...) -> i32 {
        ap.arg()
    }

    assert_eq!(unsafe { variadic(1) }, 1);
    assert_eq!(unsafe { variadic(3, 2, 1) }, 3);

    const {
        assert!(unsafe { variadic(1) } == 1);
        assert!(unsafe { variadic(3, 2, 1) } == 3);
    }
}

fn forward_by_val() {
    const unsafe fn helper(mut ap: VaList) -> i32 {
        ap.arg()
    }

    const unsafe extern "C" fn variadic(ap: ...) -> i32 {
        helper(ap)
    }

    assert_eq!(unsafe { variadic(1) }, 1);
    assert_eq!(unsafe { variadic(3, 2, 1) }, 3);

    const {
        assert!(unsafe { variadic(1) } == 1);
        assert!(unsafe { variadic(3, 2, 1) } == 3);
    }
}

fn forward_by_ref() {
    const unsafe fn helper(ap: &mut VaList) -> i32 {
        ap.arg()
    }

    const unsafe extern "C" fn variadic(mut ap: ...) -> i32 {
        helper(&mut ap)
    }

    assert_eq!(unsafe { variadic(1) }, 1);
    assert_eq!(unsafe { variadic(3, 2, 1) }, 3);

    const {
        assert!(unsafe { variadic(1) } == 1);
        assert!(unsafe { variadic(3, 2, 1) } == 3);
    }
}

#[allow(improper_ctypes_definitions)]
fn nested() {
    const unsafe fn helper(mut ap1: VaList, mut ap2: VaList) -> (i32, i32) {
        (ap1.arg(), ap2.arg())
    }

    const unsafe extern "C" fn variadic2(ap1: VaList, ap2: ...) -> (i32, i32) {
        helper(ap1, ap2)
    }

    const unsafe extern "C" fn variadic1(ap1: ...) -> (i32, i32) {
        variadic2(ap1, 2, 2)
    }

    assert_eq!(unsafe { variadic1(1) }, (1, 2));

    const {
        let (a, b) = unsafe { variadic1(1, 1) };

        assert!(a != 2);
        assert!(a == 1);
        assert!(b != 1);
        assert!(b == 2);
    }
}

fn various_types() {
    const unsafe extern "C" fn check_list_2(mut ap: ...) {
        macro_rules! continue_if {
            ($cond:expr) => {
                if !($cond) {
                    panic!();
                }
            };
        }

        const unsafe fn compare_c_str(ptr: *const c_char, val: &str) -> bool {
            match CStr::from_ptr(ptr).to_str() {
                Ok(cstr) => cstr == val,
                Err(_) => panic!(),
            }
        }

        continue_if!(ap.arg::<c_double>().floor() == 3.14f64.floor());
        continue_if!(ap.arg::<c_long>() == 12);
        continue_if!(ap.arg::<c_int>() == 'a' as c_int);
        continue_if!(ap.arg::<c_double>().floor() == 6.18f64.floor());
        continue_if!(compare_c_str(ap.arg::<*const c_char>(), "Hello"));
        continue_if!(ap.arg::<c_int>() == 42);
        continue_if!(compare_c_str(ap.arg::<*const c_char>(), "World"));
    }

    unsafe {
        check_list_2(
            3.14 as c_double,
            12 as c_long,
            b'a' as c_int,
            6.28 as c_double,
            c"Hello".as_ptr(),
            42 as c_int,
            c"World".as_ptr(),
        );
        const {
            check_list_2(
                3.14 as c_double,
                12 as c_long,
                b'a' as c_int,
                6.28 as c_double,
                c"Hello".as_ptr(),
                42 as c_int,
                c"World".as_ptr(),
            )
        };
    }
}

fn main() {
    ignores_arguments();
    echo();
    forward_by_val();
    forward_by_ref();
    nested();
    various_types();
}
