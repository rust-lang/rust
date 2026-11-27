//@ run-pass
//@ compile-flags: -lstatic=wronglibrary:rust_test_helpers

#[link(name = "wronglibrary", kind = "dylib")]
extern "C" {
    pub fn rust_dbg_extern_identity_u32(x: u32) -> u32;
}

fn main() {
    unsafe {
        rust_dbg_extern_identity_u32(42);
    }
}
