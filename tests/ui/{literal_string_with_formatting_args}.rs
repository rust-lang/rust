// Regression test for <https://github.com/rust-lang/rust-clippy/issues/13885>.
// The `dbg` macro generates a literal with the name of the current file, so
// we need to ensure the lint is not emitted in this case.

#![crate_name = "foo"]
#![allow(unused)]
#![warn(clippy::literal_string_with_formatting_args)]

fn another_bad() {
    let literal_string_with_formatting_args = 0;
    dbg!("something");
}

fn main() {}
