//@ build-pass

// Ensures that we don't regress on "implementation is not general enough" when
// normalizating under binders. Unlike `normalization-generality.rs`, this also produces
// type outlives predicates that we must ignore.

pub unsafe trait Yokeable<'a> {
    type Output: 'a;
}
pub struct Yoke<Y: for<'a> Yokeable<'a>> {
    _marker: std::marker::PhantomData<Y>,
}
impl<Y: for<'a> Yokeable<'a>> Yoke<Y> {
    pub fn project<P>(
        &self,
        _f: for<'a> fn(&<Y as Yokeable<'a>>::Output, &'a ()) -> <P as Yokeable<'a>>::Output,
    ) -> Yoke<P>
    where
        P: for<'a> Yokeable<'a>,
    {
        unimplemented!()
    }
}
pub fn slice(y: Yoke<&'static str>) -> Yoke<&'static [u8]> {
    y.project(move |yk, _| yk.as_bytes())
}
unsafe impl<'a, T: 'static + ?Sized> Yokeable<'a> for &'static T {
    type Output = &'a T;
}
fn main() {}
