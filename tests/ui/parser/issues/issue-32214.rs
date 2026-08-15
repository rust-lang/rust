trait Trait<T> { type Item; }

pub fn test<W, I: Trait<Item=(), W> >() {}
//~^ ERROR generic arguments must come before the first constraint

fn main() { }
