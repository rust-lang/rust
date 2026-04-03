#![crate_name = "anonexternmod"]

#[cfg_attr(target_env = "pauthtest", link(name = "rust_test_helpers", kind = "dylib"))]
#[cfg_attr(not(target_env = "pauthtest"), link(name = "rust_test_helpers", kind = "static"))]
extern "C" {
    pub fn rust_get_test_int() -> isize;
}
