// regression test for #74868

//@ check-pass

trait BoxedDsl<'a> {
    type Output;
}

impl<'a, T> BoxedDsl<'a> for T
where
    T: BoxedDsl<'a>,
{
    type Output = <T as BoxedDsl<'a>>::Output;
}

// Showing this trait is wf requires proving
// Self: HandleUpdate
//
// The impl below is a candidate for this projection, as well as the `Self:
// HandleUpdate` bound in the environment.
// We evaluate both candidates to see if we need to consider both applicable.
// Evaluating the impl candidate requires evaluating
// <T as BoxedDsl<'static>>::Output == ()
// The above impl cause normalizing the above type normalize to itself.
//
// This previously compiled because we would generate a new region
// variable each time around the cycle, and evaluation would eventually return
// `EvaluatedToErr` from the `Self: Sized` in the impl, which would in turn
// leave the bound as the only candidate.
//
// #73452 changed this so that region variables are canonicalized when we
// normalize, which means that the projection cycle is detected before
// evaluation returns EvaluatedToErr. The cycle resulted in an error being
// emitted immediately, causing this to fail to compile.
//
// To fix this, normalization doesn't directly emit errors when it finds a
// cycle, instead letting the caller handle it. This restores the original
// behavior.
trait HandleUpdate {}

impl<T> HandleUpdate for T where T: BoxedDsl<'static, Output = ()> {}

fn main() {}
