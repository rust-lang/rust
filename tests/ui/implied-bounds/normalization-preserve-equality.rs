// Both revisions should pass. `borrowck` revision is a bug!
//
//@ revisions: wfcheck borrowck
//@ [wfcheck] check-pass
//@ [borrowck] check-fail
//@ [borrowck] known-bug: #106569

struct Equal<'a, 'b>(&'a &'b (), &'b &'a ()); // implies 'a == 'b

trait Trait {
    type Ty;
}

impl<'x> Trait for Equal<'x, 'x> {
    type Ty = ();
}

trait WfCheckTrait {}

#[cfg(wfcheck)]
impl<'a, 'b> WfCheckTrait for (<Equal<'a, 'b> as Trait>::Ty, Equal<'a, 'b>) {}

#[cfg(borrowck)]
fn test_borrowck<'a, 'b>(_: (<Equal<'a, 'b> as Trait>::Ty, Equal<'a, 'b>)) {
    let _ = None::<Equal<'a, 'b>>;
}

fn main() {}
