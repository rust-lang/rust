//@ build-pass
//@ compile-flags: -Zdefault-hidden-visibility=yes

#![crate_type = "dylib"]

pub fn do_memcmp(left: &[u8], right: &[u8]) -> i32 {
    left.cmp(right) as i32
}
