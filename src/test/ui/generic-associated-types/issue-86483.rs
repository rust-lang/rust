// Regression test of #86483.
//
// Made to pass as part of fixing #98095.
//
// check-pass

#![feature(generic_associated_types)]

pub trait IceIce<T>
where
    for<'a> T: 'a,
{
    type Ice<'v>: IntoIterator<Item = &'v T>;
}

fn main() {}
