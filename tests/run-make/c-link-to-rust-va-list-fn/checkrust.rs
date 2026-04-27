#![crate_type = "staticlib"]
#![feature(c_variadic, c_variadic_int128)]

use core::ffi::{CStr, VaList, c_char, c_double, c_int, c_long, c_longlong};

macro_rules! continue_if {
    ($cond:expr) => {
        if !($cond) {
            return 0xff;
        }
    };
}

unsafe fn compare_c_str(ptr: *const c_char, val: &CStr) -> bool {
    val == CStr::from_ptr(ptr)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn check_list_0(mut ap: VaList) -> usize {
    continue_if!(ap.next_arg::<c_longlong>() == 1);
    continue_if!(ap.next_arg::<c_int>() == 2);
    continue_if!(ap.next_arg::<c_longlong>() == 3);
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn check_list_1(mut ap: VaList) -> usize {
    continue_if!(ap.next_arg::<c_int>() == -1);
    continue_if!(ap.next_arg::<c_int>() == 'A' as c_int);
    continue_if!(ap.next_arg::<c_int>() == '4' as c_int);
    continue_if!(ap.next_arg::<c_int>() == ';' as c_int);
    continue_if!(ap.next_arg::<c_int>() == 0x32);
    continue_if!(ap.next_arg::<i32>() == 0x10000001);
    continue_if!(compare_c_str(ap.next_arg::<*const c_char>(), c"Valid!"));
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn check_list_2(mut ap: VaList) -> usize {
    continue_if!(ap.next_arg::<c_double>() == 3.14);
    continue_if!(ap.next_arg::<c_long>() == 12);
    continue_if!(ap.next_arg::<c_int>() == 'a' as c_int);
    continue_if!(ap.next_arg::<c_double>() == 6.28);
    continue_if!(compare_c_str(ap.next_arg::<*const c_char>(), c"Hello"));
    continue_if!(ap.next_arg::<c_int>() == 42);
    continue_if!(compare_c_str(ap.next_arg::<*const c_char>(), c"World"));
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn check_list_copy_0(mut ap: VaList) -> usize {
    continue_if!(ap.next_arg::<c_double>() == 6.28);
    continue_if!(ap.next_arg::<c_int>() == 16);
    continue_if!(ap.next_arg::<c_int>() == 'A' as c_int);
    continue_if!(compare_c_str(ap.next_arg::<*const c_char>(), c"Skip Me!"));
    let mut ap = ap.clone();
    if compare_c_str(ap.next_arg::<*const c_char>(), c"Correct") { 0 } else { 0xff }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn check_list_i128(mut ap: VaList) -> usize {
    cfg_select! {
        any(
            target_arch = "aarch64",
            target_arch = "amdgpu",
            target_arch = "arm64ec",
            target_arch = "bpf",
            target_arch = "loongarch64",
            target_arch = "mips64",
            target_arch = "mips64r6",
            target_arch = "nvptx64",
            target_arch = "powerpc64",
            target_arch = "riscv64",
            target_arch = "s390x",
            target_arch = "sparc64",
            target_arch = "wasm32",
            target_arch = "wasm64",
            target_arch = "x86_64",
        ) => {
            #[cfg(not(any(
                target_arch = "wasm32",
                target_abi = "x32",
                target_pointer_width = "64",
            )))]
            compile_error!("unexpected target architecture for 128-bit c-variadic");

            continue_if!(ap.next_arg::<i128>() == -42);
            // use a 32-bit value here to test the alignment logic.
            continue_if!(ap.next_arg::<c_int>() == 0xAAAA_AAAAu32.cast_signed());
            continue_if!(ap.next_arg::<u128>() == u128::MAX);

            return 0;
        }
        _ => {
            // This function was called a platform where rustc does not implement
            // VaArgSafe for i128 but clang does define __int128. Rustc should add
            // the implementation if this comes up.
            0xFF
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn check_varargs_0(_: c_int, mut ap: ...) -> usize {
    continue_if!(ap.next_arg::<c_int>() == 42);
    continue_if!(compare_c_str(ap.next_arg::<*const c_char>(), c"Hello, World!"));
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn check_varargs_1(_: c_int, mut ap: ...) -> usize {
    continue_if!(ap.next_arg::<c_double>() == 3.14);
    continue_if!(ap.next_arg::<c_long>() == 12);
    continue_if!(ap.next_arg::<c_int>() == 'A' as c_int);
    continue_if!(ap.next_arg::<c_longlong>() == 1);
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn check_varargs_2(_: c_int, _ap: ...) -> usize {
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn check_varargs_3(_: c_int, mut ap: ...) -> usize {
    continue_if!(ap.next_arg::<c_int>() == 1);
    continue_if!(ap.next_arg::<c_int>() == 2);
    continue_if!(ap.next_arg::<c_int>() == 3);
    continue_if!(ap.next_arg::<c_int>() == 4);
    continue_if!(ap.next_arg::<c_int>() == 5);
    continue_if!(ap.next_arg::<c_int>() == 6);
    continue_if!(ap.next_arg::<c_int>() == 7);
    continue_if!(ap.next_arg::<c_int>() == 8);
    continue_if!(ap.next_arg::<c_int>() == 9);
    continue_if!(ap.next_arg::<c_int>() == 10);
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn check_varargs_4(_: c_double, mut ap: ...) -> usize {
    continue_if!(ap.next_arg::<c_double>() == 1.0);
    continue_if!(ap.next_arg::<c_double>() == 2.0);
    continue_if!(ap.next_arg::<c_double>() == 3.0);
    continue_if!(ap.next_arg::<c_double>() == 4.0);
    continue_if!(ap.next_arg::<c_double>() == 5.0);
    continue_if!(ap.next_arg::<c_double>() == 6.0);
    continue_if!(ap.next_arg::<c_double>() == 7.0);
    continue_if!(ap.next_arg::<c_double>() == 8.0);
    continue_if!(ap.next_arg::<c_double>() == 9.0);
    continue_if!(ap.next_arg::<c_double>() == 10.0);
    continue_if!(ap.next_arg::<c_double>() == 11.0);
    continue_if!(ap.next_arg::<c_double>() == 12.0);
    continue_if!(ap.next_arg::<c_double>() == 13.0);
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn check_varargs_5(_: c_int, mut ap: ...) -> usize {
    continue_if!(ap.next_arg::<c_double>() == 1.0);
    continue_if!(ap.next_arg::<c_int>() == 1);
    continue_if!(ap.next_arg::<c_double>() == 2.0);
    continue_if!(ap.next_arg::<c_int>() == 2);
    continue_if!(ap.next_arg::<c_double>() == 3.0);
    continue_if!(ap.next_arg::<c_int>() == 3);
    continue_if!(ap.next_arg::<c_double>() == 4.0);
    continue_if!(ap.next_arg::<c_int>() == 4);
    continue_if!(ap.next_arg::<c_int>() == 5);
    continue_if!(ap.next_arg::<c_double>() == 5.0);
    continue_if!(ap.next_arg::<c_int>() == 6);
    continue_if!(ap.next_arg::<c_double>() == 6.0);
    continue_if!(ap.next_arg::<c_int>() == 7);
    continue_if!(ap.next_arg::<c_double>() == 7.0);
    continue_if!(ap.next_arg::<c_int>() == 8);
    continue_if!(ap.next_arg::<c_double>() == 8.0);
    continue_if!(ap.next_arg::<c_int>() == 9);
    continue_if!(ap.next_arg::<c_double>() == 9.0);
    continue_if!(ap.next_arg::<c_int>() == 10);
    continue_if!(ap.next_arg::<c_double>() == 10.0);
    continue_if!(ap.next_arg::<c_int>() == 11);
    continue_if!(ap.next_arg::<c_double>() == 11.0);
    continue_if!(ap.next_arg::<c_int>() == 12);
    continue_if!(ap.next_arg::<c_double>() == 12.0);
    continue_if!(ap.next_arg::<c_int>() == 13);
    continue_if!(ap.next_arg::<c_double>() == 13.0);
    0
}

unsafe extern "C" {
    fn test_variadic(_: c_int, ...) -> usize;
    fn test_va_list_by_value(_: VaList) -> usize;
    fn test_va_list_by_pointer(_: *mut VaList) -> usize;
    fn test_va_list_by_pointer_pointer(_: *mut *mut VaList) -> usize;
}

#[unsafe(no_mangle)]
extern "C" fn run_test_variadic() -> usize {
    return unsafe { test_variadic(0, 1 as c_longlong, 2 as c_int, 3 as c_longlong) };
}

#[unsafe(no_mangle)]
extern "C" fn run_test_va_list_by_value() -> usize {
    unsafe extern "C" fn helper(ap: ...) -> usize {
        unsafe { test_va_list_by_value(ap) }
    }

    unsafe { helper(1 as c_longlong, 2 as c_int, 3 as c_longlong) }
}

#[unsafe(no_mangle)]
extern "C" fn run_test_va_list_by_pointer() -> usize {
    unsafe extern "C" fn helper(mut ap: ...) -> usize {
        unsafe { test_va_list_by_pointer(&mut ap) }
    }

    unsafe { helper(1 as c_longlong, 2 as c_int, 3 as c_longlong) }
}

#[unsafe(no_mangle)]
extern "C" fn run_test_va_list_by_pointer_pointer() -> usize {
    unsafe extern "C" fn helper(mut ap: ...) -> usize {
        unsafe { test_va_list_by_pointer_pointer(&mut (&mut ap as *mut _)) }
    }

    unsafe { helper(1 as c_longlong, 2 as c_int, 3 as c_longlong) }
}
