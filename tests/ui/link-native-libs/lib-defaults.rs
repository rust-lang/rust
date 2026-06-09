//! By default, `-l NAME` without an explicit kind will default to dylib. However, if there's also
//! an `#[link(name = NAME, kind = KIND)]` attribute with an explicit `KIND`, it should override the
//! CLI flag. In particular, this should not result in any duplicate flag warnings from the linker.

//@ run-pass
//@ compile-flags: -lrust_test_helpers

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    pub fn rust_dbg_extern_identity_u32(x: u32) -> u32;
}

fn main() {
    unsafe {
        rust_dbg_extern_identity_u32(42);
    }
}
