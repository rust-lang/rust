// ignore-wasm32-bare no libc to test ffi with

#[link(name = "rust_test_helpers", kind = "static")]
extern {
    fn rust_interesting_average(_: u64, ...) -> f64;
}

pub fn main() {
    // Call without variadic arguments
    unsafe {
        assert!(rust_interesting_average(0).is_nan());
    }

    // Call with direct arguments
    unsafe {
        assert_eq!(rust_interesting_average(1, 10i64, 10.0f64) as i64, 20);
    }

    // Call with named arguments, variable number of them
    let (x1, x2, x3, x4) = (10i64, 10.0f64, 20i64, 20.0f64);
    unsafe {
        assert_eq!(rust_interesting_average(2, x1, x2, x3, x4) as i64, 30);
    }

    // A function that takes a function pointer
    unsafe fn call(fp: unsafe extern fn(u64, ...) -> f64) {
        let (x1, x2, x3, x4) = (10i64, 10.0f64, 20i64, 20.0f64);
        assert_eq!(fp(2, x1, x2, x3, x4) as i64, 30);
    }

    unsafe {
        call(rust_interesting_average);

        // Make a function pointer, pass indirectly
        let x: unsafe extern fn(u64, ...) -> f64 = rust_interesting_average;
        call(x);
    }
}
