#![crate_type = "rlib"]

extern crate leaf;

pub static BAR: &str = leaf::FOO;

pub use leaf::simple;
pub use leaf::inlined;

pub fn do_a_thing() {
    let s = leaf::GenericStruct {
        a: "hello",
        b: Some("world")
    };

    let u = leaf::GenericStruct {
        a: leaf::Plain { a: 1, b: 2 },
        b: None
    };

    leaf::generic(s);
    leaf::generic(u);
}
