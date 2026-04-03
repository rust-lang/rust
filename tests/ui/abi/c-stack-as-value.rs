//@ run-pass

mod rustrt {
    #[cfg_attr(target_env = "pauthtest", link(name = "rust_test_helpers", kind = "dylib"))]
    #[cfg_attr(not(target_env = "pauthtest"), link(name = "rust_test_helpers", kind = "static"))]
    extern "C" {
        pub fn rust_get_test_int() -> isize;
    }
}

pub fn main() {
    let _foo = rustrt::rust_get_test_int;
}
