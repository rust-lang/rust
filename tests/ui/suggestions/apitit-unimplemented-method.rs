//@ aux-build:dep.rs

extern crate dep;
use dep::*;

struct Local;

impl Trait for Local {}
//~^ ERROR not all trait items implemented
//~| HELP implement the missing item: `fn foo(_: impl Sized) { todo!() }`
//~| HELP implement the missing item: `fn bar<T>(_: impl Sized) where Foo<T>: MetaSized { todo!() }`
//~| HELP implement the missing item: `fn baz<const N: usize>() { todo!() }`
//~| HELP implement the missing item: `fn quux<'a: 'b, 'b, T>() where T: ?Sized { todo!() }`

fn main() {}
