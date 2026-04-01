//@ run-pass
//@ ignore-backends: gcc
#![feature(c_variadic)]

// In rust (and C23 and above) `...` can be the only argument.
unsafe extern "C" fn only_dot_dot_dot(mut ap: ...) -> i32 {
    unsafe { ap.arg() }
}

unsafe extern "C-unwind" fn abi_c_unwind(mut ap: ...) -> i32 {
    unsafe { ap.arg() }
}

#[allow(improper_ctypes_definitions)]
unsafe extern "C" fn mix_int_float(mut ap: ...) -> (i64, f64, *const i32, f64) {
    (ap.arg(), ap.arg(), ap.arg(), ap.arg())
}

fn main() {
    unsafe {
        assert_eq!(only_dot_dot_dot(32), 32);
        assert_eq!(abi_c_unwind(32), 32);

        // Passing more arguments than expected is allowed.
        assert_eq!(only_dot_dot_dot(32, 1i64, core::ptr::null::<i32>(), 3.14f64), 32);

        let ptr = &14i32 as *const i32;
        assert_eq!(mix_int_float(12i64, 13.0f64, ptr, 15.0f64), (12, 13.0, ptr, 15.0));
    }
}
