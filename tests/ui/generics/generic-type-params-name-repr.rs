//@ dont-require-annotations: NOTE

use std::marker;

struct A;
struct B;
struct C;
struct Foo<T = A, U = B, V = C>(marker::PhantomData<(T,U,V)>);

struct Hash<T>(marker::PhantomData<T>);
struct HashMap<K, V, H = Hash<K>>(marker::PhantomData<(K,V,H)>);

fn main() {
    // Ensure that the printed type doesn't include the default type params...
    let _: Foo<isize> = ();
    //~^ ERROR mismatched types
    //~| NOTE expected `Foo<isize>`, found `()`
    //~| NOTE expected struct `Foo<isize>`
    //~| NOTE found unit type `()`

    // ...even when they're present, but the same types as the defaults.
    let _: Foo<isize, B, C> = ();
    //~^ ERROR mismatched types
    //~| NOTE expected `Foo<isize>`, found `()`
    //~| NOTE expected struct `Foo<isize>`
    //~| NOTE found unit type `()`

    // Including cases where the default is using previous type params.
    let _: HashMap<String, isize> = ();
    //~^ ERROR mismatched types
    //~| NOTE expected `HashMap<String, isize>`, found `()`
    //~| NOTE expected struct `HashMap<String, isize>`
    //~| NOTE found unit type `()`
    let _: HashMap<String, isize, Hash<String>> = ();
    //~^ ERROR mismatched types
    //~| NOTE expected `HashMap<String, isize>`, found `()`
    //~| NOTE expected struct `HashMap<String, isize>`
    //~| NOTE found unit type `()`

    // But not when there's a different type in between.
    let _: Foo<A, isize, C> = ();
    //~^ ERROR mismatched types
    //~| NOTE expected `Foo<A, isize>`, found `()`
    //~| NOTE expected struct `Foo<A, isize>`
    //~| NOTE found unit type `()`

    // And don't print <> at all when there's just defaults.
    let _: Foo<A, B, C> = ();
    //~^ ERROR mismatched types
    //~| NOTE expected `Foo`, found `()`
    //~| NOTE expected struct `Foo`
    //~| NOTE found unit type `()`
}
