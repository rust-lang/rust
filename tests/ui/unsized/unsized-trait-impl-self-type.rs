// Test sized-ness checking in substitution in impls.

// impl - struct
trait T3<Z: ?Sized> {
    fn foo(&self, z: &Z);
}

struct S5<Y>(Y);

impl<X: ?Sized> T3<X> for S5<X> {
    //~^ ERROR the size for values of type
    //~| ERROR not all trait items implemented
}

fn main() { }
