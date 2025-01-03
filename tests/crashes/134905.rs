//@ known-bug: #134905

trait Iterate<'a> {
    type Ty: Valid;
}
impl<'a, T> Iterate<'a> for T
where
    T: Check,
{
    default type Ty = ();
}

trait Check {}
impl<'a, T> Eq for T where <T as Iterate<'a>>::Ty: Valid {}

trait Valid {}
