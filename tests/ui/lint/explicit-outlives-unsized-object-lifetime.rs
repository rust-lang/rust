//@ check-pass
// Regression test for https://github.com/rust-lang/rust/issues/134902
//
// `explicit_outlives_requirements` must not suggest removing `T: 'a` from a `?Sized`
// type param. Such a bound sets the object lifetime default for `dyn Trait` uses of
// that struct (RFC 599): removing it silently changes `Struct<dyn Trait>` from
// implying `dyn Trait + 'a` to implying `dyn Trait + 'static`, breaking callers.

#![deny(explicit_outlives_requirements)]
#![allow(unused)]

trait Test {}

// Inline bound: `T: 'a + ?Sized`.
struct Ref<'a, T: 'a + ?Sized> {
    r: &'a T,
}

// Where clause for the outlives bound, inline `?Sized`.
struct Ref2<'a, T: ?Sized>
where
    T: 'a,
{
    r: &'a T,
}

// Where clause for both.
struct Ref3<'a, T>
where
    T: 'a + ?Sized,
{
    r: &'a T,
}

// Two lifetime params; neither bound should be removed.
struct Ref4<'a, 'b, T: 'a + 'b + ?Sized> {
    a: &'a T,
    b: &'b T,
}

// Verify the semantics that motivated the fix: a struct field typed `Ref<dyn Test>`
// (without an explicit lifetime) must resolve to `dyn Test + 'a`, not `dyn Test + 'static`.
struct Container<'a> {
    t: Ref<'a, dyn Test>,
}

fn check<'a>(t: Ref<'a, dyn Test + 'a>, mut c: Container<'a>) {
    c.t = t;
}

fn main() {}
