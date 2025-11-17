//@ check-pass
#![feature(pin_ergonomics)]
#![allow(dead_code, incomplete_features)]

// Handle the case where there's ambiguity between pin as a contextual keyword and pin as a path.

struct Foo;

mod pin {
    pub struct Foo;
    #[expect(non_camel_case_types)]
    pub struct pin;

    fn foo() {
        let _x: &pin = &pin;
    }
}

fn main() {
    let _x: &pin::Foo = &pin::Foo;
    let &pin: &i32 = &0;
    let ref pin: i32 = 0;
}
