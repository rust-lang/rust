// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_shorthand_field_patterns)]

#![feature(box_patterns)]
#![feature(box_syntax)]

struct Foo {a: isize, b: usize}

enum bar { u(Box<Foo>), w(isize), }

pub fn main() {
    assert!(match bar::u(box Foo{a: 10, b: 40}) {
              bar::u(box Foo{a: a, b: b}) => { a + (b as isize) }
              _ => { 66 }
            } == 50);
}
