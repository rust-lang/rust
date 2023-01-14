#![warn(unused)]
#![deny(warnings)]

struct Inv<'a>(&'a mut &'a ());

trait Trait<'a> {}
impl<'b> Trait<'b> for for<'a> fn(Inv<'a>) {}


fn with_bound()
where
    for<'b> (for<'a> fn(Inv<'a>)): Trait<'b>, //~ ERROR unnecessary parentheses around type
{}

fn main() {
    with_bound();
}
