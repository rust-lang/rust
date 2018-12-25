// compile-flags: -Z parse-only -Z continue-parse-after-error

pub fn test<W, I: Iterator<Item=(), W> >() {}
//~^ ERROR type parameters must be declared prior to associated type bindings

fn main() { }
