#![deny(dead_code)]

trait Tr {
    type A;
    type B;
    type C: Default;
    type D;
    type E;
    type F;
    type G;
    type H; //~ ERROR associated type `H` is never used
}

impl Tr for i32 {
    type A = Self;
    type B = Self;
    type C = Self;
    type D = Self;
    type E = Self;
    type F = Self;
    type G = Self;
    type H = Self;
}

trait Tr2 {
    type A;
}

impl Tr2 for i32 {
    type A = Self;
}

struct Foo<T: Tr2> {
    _x: T::A,
}

struct Bar<T> {
    _x: T
}

impl<T> Bar<T>
where
    T: Tr2<A = i32>
{}

type Baz<T> = <T as Tr>::G;

struct FooBaz<T: Tr> {
    _x: Baz<T>,
}

fn foo<T: Tr>(t: impl Tr<A = T>) -> impl Tr
where
    T::B: Copy
{
    let _a: T::C = Default::default();
    baz::<T::F>();
    t
}
fn bar<T: ?Sized>() {}
fn baz<T>() {}

fn main() {
    foo::<i32>(42);
    bar::<dyn Tr2<A = i32>>();
    let _d: <i32 as Tr>::D = Default::default();
    baz::<<i32 as Tr>::E>();
    baz::<Foo<i32>>();
    baz::<Bar<i32>>();
    baz::<FooBaz<i32>>();
}
