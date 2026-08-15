// This is yet another test to ensure that only macro calls are considered as such
// by the rustdoc highlighter, in particular when named `macro_rules`.
// This is a regression test for <https://github.com/rust-lang/rust/issues/151904>.

#![crate_name = "foo"]

//@ has src/foo/macro-call-2.rs.html
//@ count - '//code/span[@class="macro"]' 2
//@ has - '//code/span[@class="macro"]' 'macro_rules!'
//@ has - '//code/span[@class="macro"]' 'r#macro_rules!'

macro_rules! r#macro_rules {
    () => {
        fn main() {}
    }
}

r#macro_rules!();
