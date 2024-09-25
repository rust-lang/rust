//@ compile-flags: -Zfmt-debug=shallow
//@ run-pass
#![feature(fmt_debug)]
#![allow(dead_code)]
#![allow(unused)]

#[derive(Debug)]
struct Foo {
    bar: u32,
    bomb: Bomb,
}

#[derive(Debug)]
enum Baz {
    Quz,
}

struct Bomb;

impl std::fmt::Debug for Bomb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        panic!()
    }
}

fn main() {
    let s = format!("Debug is '{:?}' and '{:#?}'", Foo { bar: 1, bomb: Bomb }, Baz::Quz);
    assert_eq!("Debug is 'Foo' and 'Quz'", s);

    let f = 3.0;
    let s = format_args!("{:?}{:#?}{f:?}", 1234, cfg!(fmt_debug = "shallow")).to_string();
    assert_eq!("1234true3.0", s);
}
