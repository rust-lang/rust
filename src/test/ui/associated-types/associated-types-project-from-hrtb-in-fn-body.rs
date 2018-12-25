// Check projection of an associated type out of a higher-ranked
// trait-bound in the context of a function body.

pub trait Foo<T> {
    type A;

    fn get(&self, t: T) -> Self::A;
}

fn foo<'a, I : for<'x> Foo<&'x isize>>(
    x: <I as Foo<&'a isize>>::A)
{
    let y: I::A = x;
}

fn bar<'a, 'b, I : for<'x> Foo<&'x isize>>(
    x: <I as Foo<&'a isize>>::A,
    y: <I as Foo<&'b isize>>::A,
    cond: bool)
{
    // x and y here have two distinct lifetimes:
    let z: I::A = if cond { x } else { y };
    //~^ ERROR lifetime mismatch
}

pub fn main() {}
