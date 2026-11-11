//@ compile-flags: -Z parse-crate-root-only
//@ edition: 2021

struct S<
    T: 'a + Tr, // OK
    T: Tr + 'a, // OK
    T: 'a, // OK
    T:, // OK
    T: for<'a> ?Trait, //~ ERROR `for<...>` binder not allowed with `?` trait polarity modifier
    T: Tr +, // OK
    T: ?'a, //~ ERROR `?` may only modify trait bounds, not lifetime bounds

    T: [const] Tr, // OK
    T: [const] ?Tr, //~ ERROR `[const]` trait not allowed with `?` trait polarity modifier
    T: [const] Tr + 'a, // OK
    T: [const] 'a, //~ ERROR `[const]` may only modify trait bounds, not lifetime bounds
    T: const 'a, //~ ERROR `const` may only modify trait bounds, not lifetime bounds

    T: async Tr, // OK
    T: async ?Tr, //~ ERROR `async` trait not allowed with `?` trait polarity modifier
    T: async Tr + 'a, // OK
    T: async 'a, //~ ERROR `async` may only modify trait bounds, not lifetime bounds

    T: const async Tr, // OK
    T: const async ?Tr, //~ ERROR `const async` trait not allowed with `?` trait polarity modifier
    T: const async Tr + 'a, // OK
    T: const async 'a, //~ ERROR `const` may only modify trait bounds, not lifetime bounds
>;

fn main() {}
