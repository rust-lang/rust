// compile-flags: -Z parse-only

struct X { x: isize }

impl<'a, T, 'b> X {}
//~^ ERROR lifetime parameters must be declared prior to type parameters
