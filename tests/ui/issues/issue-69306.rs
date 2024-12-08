fn main() {}

struct S0<T>(T);
impl<T> S0<T> {
    const C: S0<u8> = Self(0);
    //~^ ERROR mismatched types
    //~| ERROR mismatched types

    fn foo() {
        Self(0);
        //~^ ERROR mismatched types
    }
}

// Testing normalization.
trait Fun {
    type Out;
}
impl<T> Fun for S0<T> {
    type Out = Self;
}
trait Foo<T> {
    fn foo();
}
impl<T> Foo<T> for <S0<T> as Fun>::Out {
    fn foo() {
        Self(0); //~ ERROR mismatched types
    }
}

struct S1<T, U>(T, U);
impl<T> S1<T, u8> {
    const C: S1<u8, u8> = Self(0, 1);
    //~^ ERROR mismatched types
    //~| ERROR mismatched types
}

struct S2<T>(T);
impl<T> S2<T> {
    fn map<U>(x: U) -> S2<U> {
        Self(x)
        //~^ ERROR mismatched types
        //~| ERROR mismatched types
    }
}
