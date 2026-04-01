//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// FIXME(#149379): This should pass, but fails due to fudged expactation
// types which are potentially not well-formed or for whom the function
// where-bounds don't actually hold. And this results in weird bugs when
// later treating these expectations as if they were actually correct..

fn foo<T>(x: (T, ())) -> Box<T> {
    Box::new(x.0)
}

fn main() {
    // Uses expectation as its struct tail is sized, resulting in `(dyn Send, ())`
    let _: Box<dyn Send> = foo(((), ()));
    //~^ ERROR mismatched types
    //~| ERROR the size for values of type `dyn Send` cannot be known at compilation time
}
