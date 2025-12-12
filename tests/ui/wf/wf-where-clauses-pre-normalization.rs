// This is a test, extracted from `modcholesky`, that used to pass wfcheck because we checked
// well-formedness of where-clauses *after* normalization. We generally want to move to always check
// WF pre-normalization.

pub struct View<A>(A);
pub trait Data {
    type Elem;
}
impl<'a, A> Data for View<&'a A> {
    type Elem = A;
}

pub fn repro<'a, T>()
where
    <View<&'a T> as Data>::Elem: Sized,
    //~^ ERROR: the parameter type `T` may not live long enough
{
}

fn main() {}
