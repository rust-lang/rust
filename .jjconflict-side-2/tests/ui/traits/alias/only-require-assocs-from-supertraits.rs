//@ check-pass

#![feature(trait_alias)]

trait Foo<T> {}
trait Bar { type Assoc; }

trait Alias<T: Bar> = Foo<T>;

// Check that an alias only requires us to specify the associated types
// of the principal's supertraits. For example, we shouldn't require
// specifying the type `Assoc` on trait `Bar` just because we have some
// `T: Bar` where clause on the alias... because that makes no sense.
fn use_alias<T: Bar>(x: &dyn Alias<T>) {}

fn main() {}
