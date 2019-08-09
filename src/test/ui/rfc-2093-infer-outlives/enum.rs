#![feature(rustc_attrs)]

// Needs an explicit where clause stating outlives condition. (RFC 2093)

// Type T needs to outlive lifetime 'a.
#[rustc_outlives]
enum Foo<'a, T> { //~ ERROR rustc_outlives
    One(Bar<'a, T>)
}

// Type U needs to outlive lifetime 'b
#[rustc_outlives]
struct Bar<'b, U> { //~ ERROR rustc_outlives
    field2: &'b U
}

// Type K needs to outlive lifetime 'c.
#[rustc_outlives]
enum Ying<'c, K> { //~ ERROR rustc_outlives
    One(&'c Yang<K>)
}

struct Yang<V> {
    field2: V
}

fn main() {}
