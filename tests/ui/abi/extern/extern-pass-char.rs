//@ run-pass

// Test a function that takes/returns a u8.

#[cfg_attr(target_env = "pauthtest", link(name = "rust_test_helpers", kind = "dylib"))]
#[cfg_attr(not(target_env = "pauthtest"), link(name = "rust_test_helpers", kind = "static"))]
extern "C" {
    pub fn rust_dbg_extern_identity_u8(v: u8) -> u8;
}

pub fn main() {
    unsafe {
        assert_eq!(22, rust_dbg_extern_identity_u8(22));
    }
}
