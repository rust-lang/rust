// We used to say "ambiguous associated type" on ambiguous associated consts.
// Ensure that we now use the correct label.

#![feature(associated_const_equality)]

trait Trait0: Parent0<i32> + Parent0<u32> {}
trait Parent0<T> { const K: (); }

fn take0(_: impl Trait0<K = { () }>) {}
//~^ ERROR ambiguous associated constant `K` in bounds of `Trait0`

trait Trait1: Parent1 + Parent2 {}
trait Parent1 { const C: i32; }
trait Parent2 { const C: &'static str; }

fn take1(_: impl Trait1<C = "?">) {}
//~^ ERROR ambiguous associated constant `C` in bounds of `Trait1`

fn main() {}
