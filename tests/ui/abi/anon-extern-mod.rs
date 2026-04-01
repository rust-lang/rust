//@ run-pass

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    fn rust_get_test_int() -> isize;
}

pub fn main() {
    unsafe {
        let _ = rust_get_test_int();
    }
}
