trait Trait<T> { type Item; }

pub fn test<W, I: Trait<Item=(), W> >() {}
//~^ ERROR constraints in a path segment must come after generic arguments

fn main() { }
