//@ run-pass

#[cfg_attr(target_env = "pauthtest", link(name = "rust_test_helpers", kind = "dylib"))]
#[cfg_attr(not(target_env = "pauthtest"), link(name = "rust_test_helpers", kind = "static"))]
extern "C" {
    fn rust_get_test_int() -> isize;
}

pub fn main() {
    unsafe {
        let _ = rust_get_test_int();
    }
}
