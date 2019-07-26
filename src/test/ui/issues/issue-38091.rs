// run-pass
#![feature(specialization)]

trait Iterate<'a> {
    type Ty: Valid;
    fn iterate(self);
}
impl<'a, T> Iterate<'a> for T where T: Check {
    default type Ty = ();
    default fn iterate(self) {}
}

trait Check {}
impl<'a, T> Check for T where <T as Iterate<'a>>::Ty: Valid {}

trait Valid {}

fn main() {
    Iterate::iterate(0);
}
