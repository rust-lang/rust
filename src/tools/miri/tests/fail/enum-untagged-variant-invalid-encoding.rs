// Validity makes this fail at the wrong place.
//@compile-flags: -Zmiri-disable-validation
use std::mem;

// This enum has untagged variant idx 1, with niche_variants being 0..=2
// and niche_start being 2.
// That means the untagged variants is in the niche variant range!
// However, using the corresponding value (2+1 = 3) is not a valid encoding of this variant.
#[derive(Copy, Clone, PartialEq)]
enum Foo {
    Var1,
    Var2(bool),
    Var3,
}

fn main() {
    unsafe {
        assert!(Foo::Var2(false) == mem::transmute(0u8));
        assert!(Foo::Var2(true) == mem::transmute(1u8));
        assert!(Foo::Var1 == mem::transmute(2u8));
        assert!(Foo::Var3 == mem::transmute(4u8));

        let invalid: Foo = mem::transmute(3u8);
        assert!(matches!(invalid, Foo::Var2(_)));
        //~^ ERROR: invalid tag
    }
}
