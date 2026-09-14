//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// FIXME(#149379): This should pass, but fails due to fudged expactation
// types which are potentially not well-formed or for whom the function
// where-bounds don't actually hold. And this results in weird bugs when
// later treating these expectations as if they were actually correct..

fn id<T>(x: Box<T>) -> Box<T> {
    x
}

fn main() {
    <[_]>::into_vec(id(Box::new([0, 1, 2])));
    //~^ ERROR: the size for values of type `[{integer}]` cannot be known at compilation time
}
