//@ check-pass

#![warn(unused)]
#![deny(warnings)]

struct Inv<'a>(#[allow(dead_code)] &'a mut &'a ());

trait Trait {}
impl Trait for for<'a> fn(Inv<'a>) {}

fn with_bound()
where
    (for<'a> fn(Inv<'a>)): Trait,
{}

fn main() {
    with_bound();
}
