#![feature(rustc_attrs)]
#![allow(unused)]

#[rustc_assert_variance]
//~^ ERROR malformed `rustc_assert_variance` attribute input
//~| NOTE expected this to be a list
//~| HELP must be of the form
struct Foo1<T> {
    val: T,
}

#[rustc_assert_variance(T)]
//~^ ERROR malformed `rustc_assert_variance` attribute input
//~| NOTE expected this to be of the form
//~| HELP must be of the form
struct Foo2<T> {
    val: T,
}

#[rustc_assert_variance(T = "nonsense")]
//~^ ERROR malformed `rustc_assert_variance` attribute input
//~| NOTE valid arguments are "covariant", "invariant" or "contravariant"
//~| HELP must be of the form
struct Foo3<T> {
    val: T,
}

#[rustc_assert_variance(T = "bivariant")]
//~^ ERROR malformed `rustc_assert_variance` attribute input
//~| NOTE valid arguments are "covariant", "invariant" or "contravariant"
//~| HELP must be of the form
struct Foo4<T> {
    val: T,
}

#[rustc_assert_variance(U = "covariant")]
//~^ ERROR not a generic type parameter of
//~| HELP rustc_assert_variance arguments should be of the form
struct Foo5<T> {
    val: T,
}

#[rustc_assert_variance(T = "invariant")]
//~^ NOTE required by this annotation
struct Foo6<T> {
    val: T,
}
//~^^^ ERROR Foo6 is covariant in T; expected variance: invariant

#[rustc_assert_variance(a = "invariant")]
//~^ NOTE required by this annotation
#[rustc_assert_variance(T = "covariant")]
//~^ NOTE required by this annotation
struct Foo7<'a, T> {
    val: &'a mut T,
}
//~^^^ ERROR Foo7 is covariant in 'a; expected variance: invariant
//~| ERROR Foo7 is invariant in T; expected variance: covariant

#[rustc_assert_variance(a = "contravariant", T = "contravariant")]
//~^ NOTE required by this annotation
//~| NOTE required by this annotation
struct Foo8<'a, T> {
    val: &'a mut T,
}
//~^^^ ERROR Foo8 is covariant in 'a; expected variance: contravariant
//~| ERROR Foo8 is invariant in T; expected variance: contravariant

// The following should all work

#[rustc_assert_variance(T = "covariant")]
struct Bar1<T> {
    val: T,
}

#[rustc_assert_variance(T = "contravariant", U = "covariant")]
struct Bar2<T, U> {
    val: fn(T) -> U,
}

#[rustc_assert_variance(a = "covariant")]
#[rustc_assert_variance(T = "invariant")]
struct Bar3<'a, T> {
    val: &'a mut T,
}

fn main() {}
