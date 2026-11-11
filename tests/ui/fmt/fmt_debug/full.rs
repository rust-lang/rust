//@ compile-flags: -Zfmt-debug=full
//@ run-pass
#![feature(fmt_debug)]
#![allow(dead_code)]
#![allow(unused)]

#[derive(Debug)]
struct Foo {
    bar: u32,
}

fn main() {
    let s = format!("Still works: {:?} '{:?}'", cfg!(fmt_debug = "full"), Foo { bar: 1 });
    assert_eq!("Still works: true 'Foo { bar: 1 }'", s);
}
