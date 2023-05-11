#![warn(unused)]
#![deny(warnings)]

struct Inv<'a>(&'a mut &'a ());

trait Trait {}
impl Trait for (for<'a> fn(Inv<'a>),) {}


fn with_bound()
where
    ((for<'a> fn(Inv<'a>)),): Trait, //~ ERROR unnecessary parentheses around type
{}

fn main() {
    with_bound();
}
