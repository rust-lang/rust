//@ run-pass

// Test a call to a function that takes/returns a u64.

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    pub fn rust_dbg_extern_identity_u64(v: u64) -> u64;
}

pub fn main() {
    unsafe {
        assert_eq!(22, rust_dbg_extern_identity_u64(22));
    }
}
