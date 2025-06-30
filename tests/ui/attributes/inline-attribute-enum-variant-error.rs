//! Test that #[inline] attribute cannot be applied to enum variants

enum Foo {
    #[inline]
    //~^ ERROR attribute should be applied
    Variant,
}

fn main() {}
