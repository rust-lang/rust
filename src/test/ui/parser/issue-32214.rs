// compile-flags: -Z continue-parse-after-error

trait Trait<T> { type Item; }

pub fn test<W, I: Trait<Item=(), W> >() {}
//~^ ERROR type parameters must be declared prior to associated type bindings

fn main() { }
