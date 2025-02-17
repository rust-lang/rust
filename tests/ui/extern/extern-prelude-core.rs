//@ run-pass
#![feature(lang_items)]
#![no_std]

extern crate std as other;

mod foo {
    pub fn test() {
        let x = core::cmp::min(2, 3);
        assert_eq!(x, 2);
    }
}

fn main() {
    foo::test();
}
