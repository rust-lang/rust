pub trait Trait<'a> {
    type Item;
}

impl<'a> Trait<'a> for () {
    type Item = ();
}

pub fn foo<T, F>(_: T, _: F)
where T: for<'a> Trait<'a>,
      F: for<'a> FnMut(<T as Trait<'a>>::Item) {}

fn main() {
    foo((), drop)
    //~^ ERROR type mismatch in function arguments
    //~| ERROR type mismatch resolving
}
