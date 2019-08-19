// compile-flags: -Z parse-only

struct S<
    T: 'a + Tr, // OK
    T: Tr + 'a, // OK
    T: 'a, // OK
    T:, // OK
    T: ?for<'a> Trait, // OK
    T: Tr +, // OK
    T: ?'a, //~ ERROR `?` may only modify trait bounds, not lifetime bounds
>;

fn main() {}
