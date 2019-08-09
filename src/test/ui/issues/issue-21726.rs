// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// Regression test for #21726: an issue arose around the rules for
// subtyping of projection types that resulted in an unconstrained
// region, yielding region inference failures.

// pretty-expanded FIXME #23616

fn main() { }

fn foo<'a>(s: &'a str) {
    let b: B<()> = B::new(s, ());
    b.get_short();
}

trait IntoRef<'a> {
    type T: Clone;
    fn into_ref(self, _: &'a str) -> Self::T;
}

impl<'a> IntoRef<'a> for () {
    type T = &'a str;
    fn into_ref(self, s: &'a str) -> &'a str {
        s
    }
}

struct B<'a, P: IntoRef<'a>>(P::T);

impl<'a, P: IntoRef<'a>> B<'a, P> {
    fn new(s: &'a str, i: P) -> B<'a, P> {
        B(i.into_ref(s))
    }

    fn get_short(&self) -> P::T {
        self.0.clone()
    }
}
