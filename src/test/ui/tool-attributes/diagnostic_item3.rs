#![feature(rustc_attrs)]

#[rustc_diagnostic_item = "foomp"]
struct Foomp;

#[rustc_diagnostic_item = "too"]
#[rustc_diagnostic_item = "much"]  //~ ERROR multiple rustc_diagnostic_item attributes found
fn failure() {}

fn main() {}
