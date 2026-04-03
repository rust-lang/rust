//@ no-prefer-dynamic

#![crate_type = "rlib"]

#[cfg_attr(target_env = "pauthtest", link(name = "rust_test_helpers", kind = "dylib"))]
#[cfg_attr(not(target_env = "pauthtest"), link(name = "rust_test_helpers", kind = "static"))]
extern "C" {
    pub fn rust_dbg_extern_identity_u32(u: u32) -> u32;
}
