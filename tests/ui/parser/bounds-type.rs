//@ compile-flags: -Z parse-only

struct S<
    T: 'a + Tr, // OK
    T: Tr + 'a, // OK
    T: 'a, // OK
    T:, // OK
    T: ?for<'a> Trait, // OK
    T: Tr +, // OK
    T: ?'a, //~ ERROR `?` may only modify trait bounds, not lifetime bounds

    T: ~const Tr, // OK
    T: ~const ?Tr, // OK
    T: ~const Tr + 'a, // OK
    T: ~const 'a, //~ ERROR `~const` may only modify trait bounds, not lifetime bounds
    T: const 'a, //~ ERROR `const` may only modify trait bounds, not lifetime bounds
>;

fn main() {}
