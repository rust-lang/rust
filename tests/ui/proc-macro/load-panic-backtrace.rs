//@ proc-macro: test-macros.rs
//@ compile-flags: -Z proc-macro-backtrace
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "thread '.*' \(0x[[:xdigit:]]+\) panicked " -> ""
//@ normalize-stderr: "note:.*RUST_BACKTRACE=1.*\n" -> ""
//@ needs-unwind proc macro panics to report errors

#[macro_use]
extern crate test_macros;

#[derive(Panic)]
//~^ ERROR: proc-macro derive panicked
struct Foo;

fn main() {}
