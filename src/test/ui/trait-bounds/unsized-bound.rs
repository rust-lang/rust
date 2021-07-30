trait Trait<A> {}
impl<A, B> Trait<(A, B)> for (A, B) where A: ?Sized, B: ?Sized, {}
//~^ ERROR E0277
//~| ERROR E0277
impl<A, B: ?Sized, C: ?Sized> Trait<(A, B, C)> for (A, B, C) where A: ?Sized, {}
//~^ ERROR E0277
//~| ERROR E0277
//~| ERROR E0277
trait Trait2<A> {}
impl<A: ?Sized, B: ?Sized> Trait2<(A, B)> for (A, B) {}
//~^ ERROR E0277
//~| ERROR E0277
trait Trait3<A> {}
impl<A> Trait3<A> for A where A: ?Sized {}
//~^ ERROR E0277
trait Trait4<A> {}
impl<A: ?Sized> Trait4<A> for A {}
//~^ ERROR E0277
trait Trait5<A, B> {}
impl<X, Y> Trait5<X, Y> for X where X: ?Sized {}
//~^ ERROR E0277
trait Trait6<A, B> {}
impl<X: ?Sized, Y> Trait6<X, Y> for X {}
//~^ ERROR E0277
trait Trait7<A, B> {}
impl<X, Y> Trait7<X, Y> for X where Y: ?Sized {}
//~^ ERROR E0277
trait Trait8<A, B> {}
impl<X, Y: ?Sized> Trait8<X, Y> for X {}
//~^ ERROR E0277

fn main() {}
