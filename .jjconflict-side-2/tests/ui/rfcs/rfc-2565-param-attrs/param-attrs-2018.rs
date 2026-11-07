//@ edition:2018

trait Trait2015 { fn foo(#[allow(C)] i32); }
//~^ ERROR expected one of `:`, `@`, or `|`, found `)`

fn main() {}
