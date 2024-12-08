//@ run-pass

// Test a function that takes/returns a u8.

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    pub fn rust_dbg_extern_identity_u8(v: u8) -> u8;
}

pub fn main() {
    unsafe {
        assert_eq!(22, rust_dbg_extern_identity_u8(22));
    }
}
