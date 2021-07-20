// Check that nested impl Trait items work in functions with generic parameters.
// check-pass

trait Captures<'a> {}

impl<T> Captures<'_> for T {}

fn nested_assoc_type<'a: 'a, T>() -> impl Iterator<Item = impl Sized> {
    //~^ ERROR cannot resolve opaque type
    [1].iter()
}

fn nested_assoc_lifetime<'a: 'a, T>() -> impl Iterator<Item = impl Captures<'a>> {
    //~^ ERROR cannot resolve opaque type
    [1].iter()
}

fn main() {}
