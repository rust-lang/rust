// ignore-tidy-linelength

// Needs an explicit where clause stating outlives condition. (RFC 2093)

// Type T needs to outlive lifetime 'a.
enum Foo<'a, T> {

    One(Bar<'a, T>)
}

// Type U needs to outlive lifetime 'b
struct Bar<'b, U> {
    field2: &'b U //~ ERROR the parameter type `U` may not live long enough [E0309]
}



// Type K needs to outlive lifetime 'c.
enum Ying<'c, K> {
    One(&'c Yang<K>) //~ ERROR the parameter type `K` may not live long enough [E0309]
}

struct Yang<V> {
    field2: V
}

fn main() {}
