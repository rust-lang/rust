// run-pass
#![feature(extern_prelude, lang_items, start)]
#![no_std]

extern crate std as other;

mod foo {
    pub fn test() {
        let x = core::cmp::min(2, 3);
        assert_eq!(x, 2);
    }
}

#[start]
fn start(_argc: isize, _argv: *const *const u8) -> isize {
    foo::test();
    0
}
