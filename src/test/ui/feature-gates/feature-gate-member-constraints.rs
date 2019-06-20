trait Trait<'a, 'b> { }
impl<T> Trait<'_, '_> for T {}

fn foo<'a, 'b>(x: &'a u32, y: &'b u32) -> impl Trait<'a, 'b> {
    //~^ ERROR ambiguous lifetime bound
    (x, y)
}

fn main() { }
