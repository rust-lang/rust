// run-pass
// ignore-wasm32-bare no libc for ffi testing

// Test a function that takes/returns a u32.

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    pub fn rust_dbg_extern_identity_u32(v: u32) -> u32;
}

pub fn main() {
    unsafe {
        assert_eq!(22, rust_dbg_extern_identity_u32(22));
    }
}
