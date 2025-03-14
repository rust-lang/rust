// Test for normalization of projections that appear in the item bounds
// (versus those that appear directly in the input types).
//
//@ revisions: param_ty lifetime param_ty_no_compat lifetime_no_compat

//@ check-pass
//@[param_ty_no_compat] compile-flags: -Zno-implied-bounds-compat
//@[lifetime_no_compat] compile-flags: -Zno-implied-bounds-compat

pub trait Iter {
    type Item;
}

#[cfg(any(param_ty, param_ty_no_compat))]
impl<X, I> Iter for I
where
    I: IntoIterator<Item = X>,
{
    type Item = X;
}

#[cfg(any(lifetime, lifetime_no_compat))]
impl<'x, I> Iter for I
where
    I: IntoIterator<Item = &'x ()>,
{
    type Item = &'x ();
}

pub struct Map<I>(I)
where
    I: Iter,
    I::Item: 'static;

pub fn test_wfcheck<'x>(_: Map<Vec<&'x ()>>) {}

pub fn test_borrowck<'x>(_: Map<Vec<&'x ()>>, s: &'x str) -> &'static str {
    s
}

fn main() {}
