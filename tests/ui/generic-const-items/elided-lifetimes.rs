#![feature(generic_const_items)]
#![allow(incomplete_features)]

// Check that we forbid elided lifetimes inside the generics of const items.

const K<T>: () = ()
where
    &T: Copy; //~ ERROR `&` without an explicit lifetime name cannot be used here

const I<const S: &str>: &str = "";
//~^ ERROR `&` without an explicit lifetime name cannot be used here
//~| ERROR `&str` is forbidden as the type of a const generic parameter

const B<T: Trait<'_>>: () = (); //~ ERROR `'_` cannot be used here

trait Trait<'a> {}

fn main() {}
