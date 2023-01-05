// Regression test of #86483.
//
// Made to pass as part of fixing #98095.
//
// check-pass

pub trait IceIce<T>
where
    for<'a> T: 'a,
{
    type Ice<'v>: IntoIterator<Item = &'v T>;
}

fn main() {}
