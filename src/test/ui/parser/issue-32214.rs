trait Trait<T> { type Item; }

pub fn test<W, I: Trait<Item=(), W> >() {}
//~^ ERROR associated type bindings must be declared after generic parameters

fn main() { }
