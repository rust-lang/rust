struct MyType;

struct MyType1<T>(T);

trait Bar {
    type Out;
}

impl<T> MyType {
    //~^ ERROR  the type parameter `T` is not constrained
}

impl<T> MyType1<T> {
    // OK, T is used in `Foo<T>`.
}

impl<T,U> MyType1<T> {
    //~^ ERROR  the type parameter `U` is not constrained
}

impl<T,U> MyType1<T> where T: Bar<Out=U> {
    // OK, T is used in `Foo<T>`.
}

fn main() { }
