// dont-check-compiler-stderr (rust-lang/rust#54222)

// ignore-wasm32-bare no libc to test ffi with

// compile-flags: -lrust_test_helpers

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    pub fn rust_dbg_extern_identity_u32(x: u32) -> u32;
}

fn main() {
    unsafe {
        rust_dbg_extern_identity_u32(42);
    }
}
