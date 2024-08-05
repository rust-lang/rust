#![deny(dead_code)]

trait Tr {
    type X; //~ ERROR associated type `X` is never used
    type Y;
    type Z;
}

impl Tr for () {
    type X = Self;
    type Y = Self;
    type Z = Self;
}

trait Tr2 {
    type X;
}

fn foo<T: Tr>() -> impl Tr<Y = ()> where T::Z: Copy {}
fn bar<T: ?Sized>() {}

fn main() {
    foo::<()>();
    bar::<dyn Tr2<X = ()>>();
}
