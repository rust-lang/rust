#![feature(rustc_attrs)]

fn _test() -> impl Default { }

#[rustc_error]
fn main() { } //~ ERROR compilation successful
