// run-pass
// ignore-wasm32-bare no libc to test ffi with

#[link(name = "rust_test_helpers", kind = "static")]
extern {
    fn rust_interesting_average(_: i64, ...) -> f64;
}

fn test<T, U>(a: i64, b: i64, c: i64, d: i64, e: i64, f: T, g: U) -> i64 {
    unsafe {
        rust_interesting_average(6, a, a as f64,
                                    b, b as f64,
                                    c, c as f64,
                                    d, d as f64,
                                    e, e as f64,
                                    f, g) as i64
    }
}

fn main(){
    assert_eq!(test(10, 20, 30, 40, 50, 60_i64, 60.0_f64), 70);
}
