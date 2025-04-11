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
    //~| NOTE_NONVIRAL expected `Foo<isize>`, found `()`
    //~| NOTE_NONVIRAL expected struct `Foo<isize>`
    //~| NOTE_NONVIRAL found unit type `()`

    // ...even when they're present, but the same types as the defaults.
    let _: Foo<isize, B, C> = ();
    //~^ ERROR mismatched types
    //~| NOTE_NONVIRAL expected `Foo<isize>`, found `()`
    //~| NOTE_NONVIRAL expected struct `Foo<isize>`
    //~| NOTE_NONVIRAL found unit type `()`

    // Including cases where the default is using previous type params.
    let _: HashMap<String, isize> = ();
    //~^ ERROR mismatched types
    //~| NOTE_NONVIRAL expected `HashMap<String, isize>`, found `()`
    //~| NOTE_NONVIRAL expected struct `HashMap<String, isize>`
    //~| NOTE_NONVIRAL found unit type `()`
    let _: HashMap<String, isize, Hash<String>> = ();
    //~^ ERROR mismatched types
    //~| NOTE_NONVIRAL expected `HashMap<String, isize>`, found `()`
    //~| NOTE_NONVIRAL expected struct `HashMap<String, isize>`
    //~| NOTE_NONVIRAL found unit type `()`

    // But not when there's a different type in between.
    let _: Foo<A, isize, C> = ();
    //~^ ERROR mismatched types
    //~| NOTE_NONVIRAL expected `Foo<A, isize>`, found `()`
    //~| NOTE_NONVIRAL expected struct `Foo<A, isize>`
    //~| NOTE_NONVIRAL found unit type `()`

    // And don't print <> at all when there's just defaults.
    let _: Foo<A, B, C> = ();
    //~^ ERROR mismatched types
    //~| NOTE_NONVIRAL expected `Foo`, found `()`
    //~| NOTE_NONVIRAL expected struct `Foo`
    //~| NOTE_NONVIRAL found unit type `()`
}
