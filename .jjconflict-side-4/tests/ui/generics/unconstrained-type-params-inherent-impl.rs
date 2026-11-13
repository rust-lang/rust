//! Test for unconstrained type parameters in inherent implementations

struct MyType;

struct MyType1<T>(T);

trait Bar {
    type Out;
}

impl<T> MyType {
    //~^ ERROR the type parameter `T` is not constrained
    // T is completely unused - this should fail
}

impl<T> MyType1<T> {
    // OK: T is used in the self type `MyType1<T>`
}

impl<T, U> MyType1<T> {
    //~^ ERROR the type parameter `U` is not constrained
    // T is used in self type, but U is unconstrained - this should fail
}

impl<T, U> MyType1<T>
where
    T: Bar<Out = U>,
{
    // OK: T is used in self type, U is constrained through the where clause
}

fn main() {}
