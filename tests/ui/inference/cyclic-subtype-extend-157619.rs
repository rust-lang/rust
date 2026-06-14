// Regression test for #157619. A `Subtype` obligation between two inference
// variables where one is then instantiated with a type containing the other
// used to make the inferred types grow without bound, resulting in an
// overflow error whose span and message are unrelated to the actual cause.
//
// With `-Znext-solver` we now use the sub-unification table to eagerly
// detect the cyclic type when generalizing.

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

fn main() {
    let mut samples = Vec::new();
    //[current]~^ ERROR overflow evaluating whether `&_` is well-formed
    let packet_buf = Vec::new();
    samples.extend(packet_buf.iter().map(|x| (x,)));
    samples.extend(packet_buf.iter().map(|&x| (x,)));
    //[next]~^ ERROR the trait bound `Vec<(&_,)>: Extend<(_,)>` is not satisfied
}
