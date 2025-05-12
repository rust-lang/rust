#![crate_name = "anonexternmod"]

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    pub fn rust_get_test_int() -> isize;
}
