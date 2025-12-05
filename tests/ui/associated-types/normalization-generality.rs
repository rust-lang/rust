//@ build-pass

// Ensures that we don't regress on "implementation is not general enough" when
// normalizating under binders.

#![feature(no_core)]

pub trait Yokeable<'a> {
    type Output: 'a;
}

pub struct Yoke<Y: for<'a> Yokeable<'a>> {
    _yokeable: Y,
}

impl<Y: for<'a> Yokeable<'a>> Yoke<Y> {
    pub fn project<'this, P>(
        &'this self,
        _f: for<'a> fn(<Y as Yokeable<'a>>::Output, &'a ()) -> <P as Yokeable<'a>>::Output,
    ) -> Yoke<P>
    where
        P: for<'a> Yokeable<'a>,
    {
        unimplemented!()
    }
}

pub fn slice(y: Yoke<&'static ()>) -> Yoke<&'static ()> {
    y.project(move |yk, _| yk)
}

impl<'a, T> Yokeable<'a> for &'static T {
    type Output = &'a T;
}

fn main() {}
