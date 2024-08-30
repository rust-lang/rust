//@ compile-flags: -Zfmt-debug=none
//@ run-pass
#![feature(fmt_debug)]
#![allow(dead_code)]
#![allow(unused)]

#[derive(Debug)]
struct Foo {
    bar: u32,
}

#[derive(Debug)]
enum Baz {
    Quz,
}

#[cfg(fmt_debug = "full")]
compile_error!("nope");

#[cfg(fmt_debug = "none")]
struct Custom;

impl std::fmt::Debug for Custom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("custom_fmt")
    }
}

fn main() {
    let c = Custom;
    let s = format!("Debug is '{:?}', '{:#?}', and '{c:?}'", Foo { bar: 1 }, Baz::Quz);
    assert_eq!("Debug is '', '', and ''", s);

    let f = 3.0;
    let s = format_args!("{:?}x{:#?}y{f:?}", 1234, "can't debug this").to_string();
    assert_eq!("xy", s);
}
