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
    //~| expected type `Foo<isize>`
    //~| found type `()`
    //~| expected struct `Foo`, found ()

    // ...even when they're present, but the same types as the defaults.
    let _: Foo<isize, B, C> = ();
    //~^ ERROR mismatched types
    //~| expected type `Foo<isize>`
    //~| found type `()`
    //~| expected struct `Foo`, found ()

    // Including cases where the default is using previous type params.
    let _: HashMap<String, isize> = ();
    //~^ ERROR mismatched types
    //~| expected type `HashMap<std::string::String, isize>`
    //~| found type `()`
    //~| expected struct `HashMap`, found ()
    let _: HashMap<String, isize, Hash<String>> = ();
    //~^ ERROR mismatched types
    //~| expected type `HashMap<std::string::String, isize>`
    //~| found type `()`
    //~| expected struct `HashMap`, found ()

    // But not when there's a different type in between.
    let _: Foo<A, isize, C> = ();
    //~^ ERROR mismatched types
    //~| expected type `Foo<A, isize>`
    //~| found type `()`
    //~| expected struct `Foo`, found ()

    // And don't print <> at all when there's just defaults.
    let _: Foo<A, B, C> = ();
    //~^ ERROR mismatched types
    //~| expected type `Foo`
    //~| found type `()`
    //~| expected struct `Foo`, found ()
}
