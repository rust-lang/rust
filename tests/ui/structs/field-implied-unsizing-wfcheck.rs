struct FooStruct {
    nested: &'static Bar<dyn std::fmt::Debug>,
    //~^ ERROR the size for values of type `(dyn Debug + 'static)` cannot be known at compilation time
}

struct FooTuple(&'static Bar<dyn std::fmt::Debug>);
//~^ ERROR the size for values of type `(dyn Debug + 'static)` cannot be known at compilation time

enum FooEnum1 {
    Struct { nested: &'static Bar<dyn std::fmt::Debug> },
    //~^ ERROR the size for values of type `(dyn Debug + 'static)` cannot be known at compilation time
}

enum FooEnum2 {
    Tuple(&'static Bar<dyn std::fmt::Debug>),
    //~^ ERROR the size for values of type `(dyn Debug + 'static)` cannot be known at compilation time
}

struct Bar<T>(T);

fn main() {
    // Ensure there's an error at the construction site, for error tainting purposes.

    FooStruct { nested: &Bar(4) };
    //~^ ERROR the size for values of type `(dyn Debug + 'static)` cannot be known at compilation time
    FooTuple(&Bar(4));
    //~^ ERROR the size for values of type `(dyn Debug + 'static)` cannot be known at compilation time
    FooEnum1::Struct { nested: &Bar(4) };
    //~^ ERROR the size for values of type `(dyn Debug + 'static)` cannot be known at compilation time
    FooEnum2::Tuple(&Bar(4));
    //~^ ERROR the size for values of type `(dyn Debug + 'static)` cannot be known at compilation time
}
