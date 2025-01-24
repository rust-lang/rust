// regression test for https://github.com/rust-lang/rust/issues/85763

#![crate_name = "foo"]

//@ has foo/index.html '//main' 'Some text that should be concatenated.'
#[doc = " Some text"]
#[doc = r" that should"]
/// be concatenated.
pub fn main() {
    println!("Hello, world!");
}
