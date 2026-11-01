//@ build-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
// Regression test for #135462.
#![allow(coherence_leak_check)]

type A = fn(&'static ());
type B = fn(&());

trait Bound<P: WithAssoc>: From<GetAssoc<P>> {
}
impl Bound<B> for String {}

trait Trt<T> {
    fn __(&self, x: T) where T: Bound<A> {
        T::from(());
    }
}

impl<T, S> Trt<T> for S {}

type GetAssoc<T> = <T as WithAssoc>::Ty;

trait WithAssoc {
    type Ty;
}

impl WithAssoc for B {
    type Ty = String;
}

impl WithAssoc for A {
    type Ty = ();
}

fn main() {
    let x: &'static dyn Trt<String> = &();
}
