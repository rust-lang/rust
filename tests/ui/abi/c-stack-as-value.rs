//@ run-pass

mod rustrt {
    #[link(name = "rust_test_helpers", kind = "static")]
    extern "C" {
        pub fn rust_get_test_int() -> isize;
    }
}

pub fn main() {
    let _foo = rustrt::rust_get_test_int;
}
