// Test sized-ness checking in substitution in impls.

// impl - struct

struct S5<Y>(Y);

impl<X: ?Sized> S5<X> {
    //~^ ERROR the size for values of type
}

fn main() { }
