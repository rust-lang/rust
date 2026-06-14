// Regression test for #128397. Pushing a borrow of a collection's element
// back into the collection requires the element type to contain its own
// reference. This used to make the inferred type grow without bound,
// resulting in an overflow error whose span and message are unrelated to
// the actual cause.
//
// With `-Znext-solver` we now use the sub-unification table to eagerly
// detect the cyclic type when generalizing.

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

fn main() {
    let mut v = vec![];
    let x = v.last().unwrap();
    //[current]~^ ERROR overflow evaluating whether `&_` is well-formed
    v.push(x);
    //[next]~^ ERROR mismatched types
}
