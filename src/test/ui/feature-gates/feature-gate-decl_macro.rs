#![allow(unused_macros)]

macro m() {} //~ ERROR `macro` is experimental

macro_rules! accept_item { ($i:item) => {} }
accept_item! {
    macro m() {} //~ ERROR `macro` is experimental
}
fn main() {}
