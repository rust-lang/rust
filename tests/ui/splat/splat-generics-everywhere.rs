//@ run-pass
//! Test using `#[rustc_splat]` on tuples with generics in various positions.

#![allow(incomplete_features)]
#![feature(splat)]
#![feature(tuple_trait)]

struct Foo<T>(T);

impl<T> Foo<T> {
    fn new(t: T) -> Self {
        Self(t)
    }

    fn assoc<U>(_u: U, #[rustc_splat] _s: ()) {}

    fn method<V>(&self, _v: V, #[rustc_splat] _s: (u32, f64)) {}

    fn lifetime<'a>(&self, #[rustc_splat] _s: (u32, f64, &'a str)) {}

    fn const_generic<const N: usize>(&self, #[rustc_splat] _s: (u32, f64, [u8; N])) {}

    fn generic_in_tuple<U>(&self, #[rustc_splat] _s: (U, u32)) {}

    fn generic_tuple_assoc<U: std::marker::Tuple>(_u: U, #[rustc_splat] _s: ()) {}
}

trait BarTrait<T> {
    fn trait_assoc<W>(w: W, #[rustc_splat] _s: ());

    fn trait_method<X>(&self, x: X, #[rustc_splat] _s: (u32, f64));

    fn trait_lifetime<'a>(&self, #[rustc_splat] _s: (u32, f64, &'a str)) {}

    fn trait_const_generic<const N: usize>(&self, #[rustc_splat] _s: (u32, f64, [u8; N])) {}

    fn trait_generic_in_tuple<U>(&self, #[rustc_splat] _s: (T, U)) {}

    fn trait_generic_tuple<U: std::marker::Tuple>(&self, #[rustc_splat] _s: U) {}
}

impl<T> BarTrait<T> for Foo<T> {
    fn trait_assoc<W>(_w: W, #[rustc_splat] _s: ()) {}

    fn trait_method<X>(&self, _x: X, #[rustc_splat] _s: (u32, f64)) {}

    fn trait_lifetime<'a>(&self, #[rustc_splat] _s: (u32, f64, &'a str)) {}

    fn trait_const_generic<const N: usize>(&self, #[rustc_splat] _s: (u32, f64, [u8; N])) {}

    fn trait_generic_in_tuple<U>(&self, #[rustc_splat] _s: (T, U)) {}

    fn trait_generic_tuple<U: std::marker::Tuple>(&self, #[rustc_splat] _s: U) {}
}

fn main() {
    Foo::<f32>::assoc("u");
    Foo::<f32>::trait_assoc("w");

    let foo = Foo::new("t");
    foo.method("v", 1u32, 2.3);
    foo.lifetime(1u32, 2.3, "asdf");
    foo.const_generic(1u32, 2.3, [1, 2, 3]);
    foo.generic_in_tuple(42i32, 1u32);
    Foo::<f32>::generic_tuple_assoc(());

    Foo::<u32>::trait_assoc("w");
    foo.trait_method("x", 42u32, 9.8);
    foo.trait_lifetime(1u32, 2.3, "asdf");
    foo.trait_const_generic(1u32, 2.3, [1, 2, 3]);
    foo.trait_generic_in_tuple("hello", 42i32);
    foo.trait_generic_tuple();
}
