// Test sized-ness checking in substitution in impls.

// impl - unbounded
trait T2<Z> {
    fn foo(&self, z: Z);
}
struct S4<Y: ?Sized>(Box<Y>);
impl<X: ?Sized> T2<X> for S4<X> {
    //~^ ERROR the size for values of type
}

fn main() { }
