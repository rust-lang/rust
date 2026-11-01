//@ needs-asm-support
//@ pp-exact

#[cfg(foo = r#"just parse this"#)]
extern crate blah as blah;

use std::arch::asm;

fn main() { unsafe { asm!(r###"blah"###); } }
