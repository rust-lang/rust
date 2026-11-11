//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// FIXME(#149379): This should pass, but fails due to fudged expactation
// types which are potentially not well-formed or for whom the function
// where-bounds don't actually hold. And this results in weird bugs when
// later treating these expectations as if they were actually correct..

fn sized_box<T>(x: Box<T>) -> Box<T> {
    x
}

fn main() {
    let _: Box<dyn Send> = sized_box(Box::new(1));
    //~^ ERROR the size for values of type `dyn Send` cannot be known at compilation time
}
