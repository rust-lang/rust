// All the revisions should pass. `borrowck_current` revision is a bug!
//
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ revisions: wfcheck borrowck_current borrowck_next
//@ [wfcheck] check-pass
//@ [borrowck_current] check-fail
//@ [borrowck_current] known-bug: #106569
//@ [borrowck_next] compile-flags: -Znext-solver
//@ [borrowck_next] check-pass

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

#[cfg(any(borrowck_current, borrowck_next))]
fn test_borrowck<'a, 'b>(_: (<Equal<'a, 'b> as Trait>::Ty, Equal<'a, 'b>)) {
    let _ = None::<Equal<'a, 'b>>;
}

fn main() {}
