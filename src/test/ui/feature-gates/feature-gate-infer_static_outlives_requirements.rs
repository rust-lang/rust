// Needs an explicit where clause stating outlives condition. (RFC 2093)

// Type T needs to outlive lifetime 'static.
struct Foo<U> {
    bar: Bar<U> //~ ERROR the parameter type `U` may not live long enough [E0310]
}
struct Bar<T: 'static> {
    x: T,
}


fn main() { }
