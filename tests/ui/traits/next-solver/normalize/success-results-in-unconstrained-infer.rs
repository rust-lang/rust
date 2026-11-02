//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #135122

// A regression test for #135122. The old solver normalized `<Self as Add>::Output` to
// an inference variable without having any stalled or failed obligation. This caused
// us to then compute the implied bounds for that inference variable, causing an ICE.

trait Add {
    type Output;
    fn add(_: (), _: Self::Output) {}
}

trait IsSame<Lhs> {
    type Assoc;
}

trait Data {
    type Elem;
}

impl<B> IsSame<i16> for f32 where f32: IsSame<B, Assoc = B> {}
//[next]~^ ERROR: not all trait items implemented, missing: `Assoc`
//[next]~| ERROR: the type parameter `B` is not constrained by the impl trait, self type, or predicates
impl<A> Add for i64
where
    f32: IsSame<A>,
    i8: Data<Elem = A>,
    //[next]~^ ERROR: the trait bound `i8: Data` is not satisfied
{
    type Output = <f32 as IsSame<A>>::Assoc;
    fn add(_: Missing, _: Self::Output) {}
    //[next]~^ ERROR: cannot find type `Missing` in this scope
}
fn main() {}
