//@ compile-flags: -Znext-solver

trait Trait {
    type Ty;
}

impl Trait for for<'a> fn(&'a u8, &'a u8) {
    type Ty = ();
}

// argument is necessary to create universes before registering the hidden type.
fn test<'a>(_: <fn(&u8, &u8) as Trait>::Ty) -> impl Sized {
    //~^ ERROR the type `<for<'a, 'b> fn(&'a u8, &'b u8) as Trait>::Ty` is not well-formed
    "hidden type is `&'?0 str` with '?0 member of ['static,]"
}

fn main() {}
