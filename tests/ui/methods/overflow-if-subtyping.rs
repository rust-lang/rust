//@ check-pass

// Regression test for #128887.
#![allow(unconditional_recursion)]
trait Mappable<T> {
    type Output;
}

trait Bound<T> {}
// Deleting this impl made it compile on beta
impl<T> Bound<T> for T {}

trait Generic<M> {}

// Deleting the `: Mappable<T>` already made it error on stable.
struct IndexWithIter<I, M: Mappable<T>, T>(I, M, T);

impl<I, M, T> IndexWithIter<I, M, T>
where
    <M as Mappable<T>>::Output: Bound<T>,
    // Flipping these where bounds causes this to succeed, even when removing
    // the where-clause on the struct definition.
    M: Mappable<T>,
    I: Generic<M>,
{
    fn new(x: I) {
        IndexWithIter::<_, _, _>::new(x);
    }
}
fn main() {}
