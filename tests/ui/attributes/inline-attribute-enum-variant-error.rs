//! Test that #[inline] attribute cannot be applied to enum variants

enum Foo {
    #[inline]
    //~^ ERROR attribute cannot be used on
    Variant,
}

fn main() {}
