fn f() -> i32 {
    42
}

fn return_fn_ptr() -> fn() -> i32 {
    f
}

fn call_fn_ptr() -> i32 {
    return_fn_ptr()()
}

fn main() {
    assert_eq!(call_fn_ptr(), 42);
    assert!(return_fn_ptr() == f);
    assert!(return_fn_ptr() as unsafe fn() -> i32 == f as fn() -> i32 as unsafe fn() -> i32);
}
