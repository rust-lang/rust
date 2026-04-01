//@ run-pass
//@ check-run-results

#![feature(rustc_private)]
use std::io::Write;
extern crate rustc_driver;
extern crate rustc_driver_impl;

use rustc_driver_impl::highlighter::highlight;

const TEST_INPUT: &str = "
struct Foo;

fn baz(x: i32) {
    // A function
}

fn main() {
    let foo = Foo;
    foo.bar();
}
";

fn main() {
    let mut buf = Vec::new();
    highlight(TEST_INPUT, &mut buf).unwrap();
    let mut stdout = std::io::stdout();
    stdout.write_all(&buf).unwrap();
}
