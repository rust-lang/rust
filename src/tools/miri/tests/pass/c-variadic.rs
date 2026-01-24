#![feature(c_variadic)]

use std::ffi::{CStr, VaList, c_char, c_double, c_int, c_long};

fn ignores_arguments() {
    unsafe extern "C" fn variadic(_: ...) {}

    unsafe { variadic() };
    unsafe { variadic(1, 2, 3) };
}

fn echo() {
    unsafe extern "C" fn variadic(mut ap: ...) -> i32 {
        ap.arg()
    }

    assert_eq!(unsafe { variadic(1) }, 1);
    assert_eq!(unsafe { variadic(3, 2, 1) }, 3);
}

fn forward_by_val() {
    unsafe fn helper(mut ap: VaList) -> i32 {
        ap.arg()
    }

    unsafe extern "C" fn variadic(ap: ...) -> i32 {
        helper(ap)
    }

    assert_eq!(unsafe { variadic(1) }, 1);
    assert_eq!(unsafe { variadic(3, 2, 1) }, 3);
}

fn forward_by_ref() {
    unsafe fn helper(ap: &mut VaList) -> i32 {
        ap.arg()
    }

    unsafe extern "C" fn variadic(mut ap: ...) -> i32 {
        helper(&mut ap)
    }

    assert_eq!(unsafe { variadic(1) }, 1);
    assert_eq!(unsafe { variadic(3, 2, 1) }, 3);
}

#[allow(improper_ctypes_definitions)]
fn nested() {
    unsafe fn helper(mut ap1: VaList, mut ap2: VaList) -> (i32, i32) {
        (ap1.arg(), ap2.arg())
    }

    unsafe extern "C" fn variadic2(ap1: VaList, ap2: ...) -> (i32, i32) {
        helper(ap1, ap2)
    }

    unsafe extern "C" fn variadic1(ap1: ...) -> (i32, i32) {
        variadic2(ap1, 2, 2)
    }

    assert_eq!(unsafe { variadic1(1) }, (1, 2));

    let (a, b) = unsafe { variadic1(1, 1) };

    assert_eq!(a, 1);
    assert_eq!(b, 2);
}

fn various_types() {
    unsafe extern "C" fn check_list_2(mut ap: ...) {
        macro_rules! continue_if {
            ($cond:expr) => {
                if !($cond) {
                    panic!();
                }
            };
        }

        unsafe fn compare_c_str(ptr: *const c_char, val: &str) -> bool {
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
