// run-pass
// ignore-wasm32-bare no libc to test ffi with
// compile-flags: -lstatic=wronglibrary:rust_test_helpers

#[link(name = "wronglibrary", kind = "dylib")]
extern "C" {
    pub fn rust_dbg_extern_identity_u32(x: u32) -> u32;
}

fn main() {
    unsafe {
        rust_dbg_extern_identity_u32(42);
    }
}
