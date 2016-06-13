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
}
