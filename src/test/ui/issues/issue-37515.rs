#![feature(rustc_attrs)]
#![warn(unused)]

type Z = for<'x> Send;
//~^ WARN type alias is never used

#[rustc_error]
fn main() { //~ ERROR compilation successful
}
