// Needs an explicit where clause stating outlives condition. (RFC 2093)

// Type T needs to outlive lifetime 'a.
struct Foo<'a, T> {
    bar: &'a [T] //~ ERROR the parameter type `T` may not live long enough [E0309]
}

fn main() { }
