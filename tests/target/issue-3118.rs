use {
    crate::foo::bar,
    bytes::{Buf, BufMut},
    std::io,
};

mod foo {
    pub mod bar {}
}

fn main() {}
