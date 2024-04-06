//@ run-pass
//@ revisions: old next
//@[next] compile-flags: -Znext-solver
#![allow(coherence_leak_check)]

trait Trait: Sized {
    fn is_higher_ranked(self) -> bool;
}

impl Trait for for<'a> fn(&'a ()) {
    fn is_higher_ranked(self) -> bool {
        true
    }
}
impl<'a> Trait for fn(&'a ()) {
    fn is_higher_ranked(self) -> bool {
        false
    }
}

fn main() {
    let x: for<'a> fn(&'a ()) = |&()| ();
    assert!(x.is_higher_ranked());
}
