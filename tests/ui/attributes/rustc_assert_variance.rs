#![feature(rustc_attrs)]
#![allow(unused)]

struct Foo1<#[rustc_assert_variance] T> {
    //~^ ERROR malformed `rustc_assert_variance` attribute input
    //~| NOTE expected this to be a list
    val: T,
}

struct Foo2<#[rustc_assert_variance(nonsense)] T> {
    //~^ ERROR malformed `rustc_assert_variance` attribute input
    //~| NOTE valid arguments are `covariant`, `invariant`, `contravariant` or `bivariant`
    val: T,
}

struct Foo3<#[rustc_assert_variance(bivariant)] T> {
    //~^ ERROR `Foo3` is expected to be bivariant in `T` but is not
    //~| NOTE `Foo3` is covariant in `T`
    //~| NOTE required by this attribute
    val: T,
}

struct Foo4<#[rustc_assert_variance(invariant)] T> {
    //~^ ERROR `Foo4` is expected to be invariant in `T` but is not
    //~| NOTE `Foo4` is covariant in `T`
    //~| NOTE required by this attribute
    val: T,
}

struct Foo5<
    //~^ ERROR `Foo5` is expected to be invariant in `'a` but is not
    //~| ERROR `Foo5` is expected to be covariant in `T` but is not
    #[rustc_assert_variance(invariant)]
    //~^ NOTE required by this attribute
    'a,
    //~^ NOTE `Foo5` is covariant in `'a`
    #[rustc_assert_variance(covariant)]
    //~^ NOTE required by this attribute
    T,
    //~^ NOTE `Foo5` is invariant in `T`
> {
    val: &'a mut T,
}

struct Foo6<
    //~^ ERROR `Foo6` is expected to be contravariant in `'a` but is not
    //~| ERROR `Foo6` is expected to be contravariant in `T` but is not
    #[rustc_assert_variance(contravariant)]
    //~^ NOTE required by this attribute
    'a,
    //~^ NOTE `Foo6` is covariant in `'a`
    #[rustc_assert_variance(contravariant)]
    //~^ NOTE required by this attribute
    T,
    //~^ NOTE `Foo6` is invariant in `T`
> {
    val: &'a mut T,
}

// The following should all work

struct Bar1<#[rustc_assert_variance(covariant)] T> {
    val: T,
}

struct Bar2<#[rustc_assert_variance(contravariant)] T, #[rustc_assert_variance(covariant)] U> {
    val: fn(T) -> U,
}

struct Bar3<#[rustc_assert_variance(covariant)] 'a, #[rustc_assert_variance(invariant)] T> {
    val: &'a mut T,
}

struct Bar4<#[rustc_assert_variance(invariant)] const N: usize> {}

struct Bar5<#[rustc_assert_variance(covariant)] T, #[rustc_assert_variance(bivariant)] U>
where
    T: Iterator<Item = U>,
{
    val: T,
}

fn main() {}
