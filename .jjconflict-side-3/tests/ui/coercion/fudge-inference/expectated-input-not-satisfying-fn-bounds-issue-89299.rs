//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// FIXME(#149379): This should pass, but fails due to fudged expactation
// types which are potentially not well-formed or for whom the function
// where-bounds don't actually hold. And this results in weird bugs when
// later treating these expectations as if they were actually correct..

use std::pin::Pin;

trait Trait {}

impl Trait for i32 {}

struct Foo<'a>(Pin<&'a mut (dyn Trait + Send)>);

fn main() {
    let mut a = 1;
    let _x = Foo(Pin::new(&mut a));
    //~^ ERROR: `dyn Trait + Send` cannot be unpinned
}
