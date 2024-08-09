//@ build-pass
//@ compile-flags: -Zdefault-hidden-visibility=yes
//@ ignore-wasm32 dylibs unsupported, and below does not work
//@ needs-dynamic-linking

#![crate_type = "dylib"]

pub fn do_memcmp(left: &[u8], right: &[u8]) -> i32 {
    left.cmp(right) as i32
}
