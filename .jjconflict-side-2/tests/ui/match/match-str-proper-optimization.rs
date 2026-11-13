//! Regression test for <https://github.com/rust-lang/rust/issues/22008>.
//! Ensure matching against `str` doesn't cause segfault when compiled
//! with `opt-level` >= 2.
//
//@ compile-flags: -C opt-level=2
//@ run-pass
pub fn main() {
    let command = "a";

    match command {
        "foo" => println!("foo"),
        _     => println!("{}", command),
    }
}
