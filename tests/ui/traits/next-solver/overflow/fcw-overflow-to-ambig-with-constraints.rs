//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Regression test for `calimero-store`

#![recursion_limit = "8"]
struct W<T, U>(T, U);
trait Count {}
impl<T: Count, U: Count> Count for W<T, U> {}
impl Count for () {}
// Old solver is able to use cache entries from lower recursion depths,
// new one correctly tracks their required depth, so it needs more than 8 steps.
type Four<T> = W<T, W<T, W<T, W<T, T>>>>;

trait Constrain<T, U, W> {}
impl<U: Clone, C: Count> Constrain<u32, U, C> for u32 {}

trait Equal<T> {}
impl<T> Equal<T> for T {}

fn fun_times<T, U>()
where
    u32: Constrain<T, U, Four<Four<Four<()>>>>,
    T: Equal<U>,
{}

fn main() {
    fun_times();
    //[next]~^ WARN overflow evaluating the requirement `u32: Constrain<_, _, W<W<W<(), W<(), _>>, _>, _>>`
    //[next]~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //[next]~| WARN  overflow evaluating the requirement `u32: Constrain<u32, u32, W<W<W<(), W<(), _>>, _>, _>>`
    //[next]~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //[next]~| WARN overflow evaluating the requirement `u32: Constrain<u32, u32, W<W<W<(), W<(), _>>, _>, _>>`
    //[next]~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}
