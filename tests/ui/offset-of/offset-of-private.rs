#![feature(offset_of)]

use std::mem::offset_of;

mod m {
    #[repr(C)]
    pub struct Foo {
        pub public: u8,
        private: u8,
    }
}

fn main() {
    offset_of!(m::Foo, public);
    offset_of!(m::Foo, private); //~ ERROR field `private` of struct `Foo` is private
}
