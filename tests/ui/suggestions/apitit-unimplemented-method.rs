//@ aux-build:dep.rs

extern crate dep;
use dep::*;

struct Local;
impl Trait for Local {}
//~^ ERROR not all trait items implemented
//~| HELP implement the missing item: `fn foo(_: impl Sized) { todo!() }`
//~| HELP implement the missing item: `fn bar<T>(_: impl Sized) { todo!() }`

fn main() {}
