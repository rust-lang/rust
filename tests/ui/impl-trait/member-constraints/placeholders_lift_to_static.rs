//@ check-pass

// We have some `RPIT` with an item bound of `for<'a> Outlives<'a>`. We
// infer a hidden type of `&'?x i32` where `'?x` is required to outlive
// some placeholder `'!a` due to the `for<'a> Outlives<'a>` item bound.
//
// We previously did not write constraints of the form `'?x: '!a` into
// `'?x: 'static`. This caused member constraints to bail and not consider
// `'?x` to be constrained to an arg region.

pub trait Outlives<'a> {}
impl<'a, T: 'a> Outlives<'a> for T {}

pub fn foo() -> impl for<'a> Outlives<'a> {
    let x: &'static i32 = &1;
    x
}

// This *didn't* regress but feels like it's "the same thing" so
// test it anyway
pub fn bar() -> impl Sized {
    let x: &'static i32 = &1;
    hr_outlives(x)
}

fn hr_outlives<T>(v: T) -> T
where
    for<'a> T: 'a
{
    v
}

fn main() {}
