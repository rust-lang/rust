//@ run-pass
//! Test using `#[splat]` on tuples with generics in various positions.

#![allow(incomplete_features)]
#![feature(splat)]

struct Foo<T>(T);

// FIXME(splat): also add assoc/method with splatted generic tuple traits
// also add generics inside the splatted tuple
impl<T> Foo<T> {
    fn new(t: T) -> Self {
        Self(t)
    }

    fn assoc<U>(_u: U, #[splat] _s: ()) {}

    fn method<V>(&self, _v: V, #[splat] _s: (u32, f64)) {}

    fn lifetime<'a>(&self, #[splat] _s: (u32, f64, &'a str)) {}

    fn const_generic<const N: usize>(&self, #[splat] _s: (u32, f64, [u8; N])) {}
}

// FIXME(splat): also add generics to the trait
// also add assoc/method with splatted generic tuple traits
// also add generics inside the splatted tuple
trait BarTrait {
    fn trait_assoc<W>(w: W, #[splat] _s: ());

    fn trait_method<X>(&self, x: X, #[splat] _s: (u32, f64));

    fn trait_lifetime<'a>(&self, #[splat] _s: (u32, f64, &'a str)) {}

    fn trait_const_generic<const N: usize>(&self, #[splat] _s: (u32, f64, [u8; N])) {}
}

impl<T> BarTrait for Foo<T> {
    fn trait_assoc<W>(_w: W, #[splat] _s: ()) {}

    fn trait_method<X>(&self, _x: X, #[splat] _s: (u32, f64)) {}

    fn trait_lifetime<'a>(&self, #[splat] _s: (u32, f64, &'a str)) {}

    fn trait_const_generic<const N: usize>(&self, #[splat] _s: (u32, f64, [u8; N])) {}
}

// FIXME(splat):
// - add `T: Tuple` generics tests
// - add const fn generics tests

fn main() {
    // FIXME(splat): should splatted functions be callable with tupled and un-tupled arguments?
    // Add a tupled test for each call if they are.
    //Foo::<i64>::assoc(("u",));

    Foo::<f32>::assoc("u");
    Foo::<f32>::trait_assoc("w");

    let foo = Foo::new("t");
    foo.method("v", 1u32, 2.3);
    foo.lifetime(1u32, 2.3, "asdf");
    foo.const_generic(1u32, 2.3, [1, 2, 3]);

    foo.trait_method("x", 42u32, 9.8);
    foo.trait_lifetime(1u32, 2.3, "asdf");
    foo.trait_const_generic(1u32, 2.3, [1, 2, 3]);
}
