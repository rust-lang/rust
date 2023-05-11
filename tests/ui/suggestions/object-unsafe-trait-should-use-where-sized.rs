// run-rustfix
#![allow(unused_variables, dead_code)]

trait Trait {
    fn foo() where Self: Other, { }
    fn bar(self: ()) {} //~ ERROR invalid `self` parameter type
}

fn bar(x: &dyn Trait) {} //~ ERROR the trait `Trait` cannot be made into an object

trait Other {}

fn main() {}
