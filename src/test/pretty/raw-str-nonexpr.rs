// pp-exact

#![feature(asm)]

#[cfg(foo = r#"just parse this"#)]
extern crate blah as blah;

fn main() { unsafe { asm!(r###"blah"###); } }
