// Test for normalization of projections that appear in the item bounds
// (versus those that appear directly in the input types).
// Both revisions should pass. `lifetime` revision is a bug.
//
// revisions: param_ty lifetime
// [param_ty] check-pass
// [lifetime] check-fail
// [lifetime] issue: #109799

pub trait Iter {
    type Item;
}

#[cfg(param_ty)]
impl<X, I> Iter for I
where
    I: IntoIterator<Item = X>,
{
    type Item = X;
}

#[cfg(lifetime)]
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

// passes for lifetime and type paramters.
pub fn test_wfcheck<'x>(_: Map<Vec<&'x ()>>) {}

pub fn test_borrowck<'x>(_: Map<Vec<&'x ()>>, s: &'x str) -> &'static str {
    s
    //[lifetime]~^ ERROR lifetime may not live long enough
}

fn main() {}
