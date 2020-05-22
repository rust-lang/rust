// pp-exact

#![feature(llvm_asm)]

#[cfg(foo = r#"just parse this"#)]
extern crate blah as blah;

fn main() { unsafe { llvm_asm!(r###"blah"###); } }
