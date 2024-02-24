//@ check-pass
// Weak alias types constrain late-bound regions if their normalized form constrains them.

#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

type Ref<'a> = &'a ();

type FnPtr = for<'a> fn(Ref<'a>) -> &'a (); // OK
type DynCl = dyn for<'a> Fn(Ref<'a>) -> &'a (); // OK

fn map0(_: Ref) -> Ref { &() } // OK
fn map1(_: Ref<'_>) -> Ref<'_> { &() } // OK

fn main() {}
