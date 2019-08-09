// run-pass
// ignore-wasm32-bare no libc for ffi testing

#[link(name = "rust_test_helpers", kind = "static")]
extern {
    pub fn rust_dbg_extern_identity_double(v: f64) -> f64;
}

pub fn main() {
    unsafe {
        assert_eq!(22.0_f64, rust_dbg_extern_identity_double(22.0_f64));
    }
}
