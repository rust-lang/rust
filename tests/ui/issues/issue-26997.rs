//@ build-pass
#![allow(dead_code)]
pub struct Foo {
    x: isize,
    y: isize
}

impl Foo {
    #[allow(improper_c_fn_definitions)]
    pub extern "C" fn foo_new() -> Foo {
        Foo { x: 21, y: 33 }
    }
}

fn main() {}
