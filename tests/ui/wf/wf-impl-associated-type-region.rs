// Check that we require that associated types in an impl are well-formed.



pub trait Foo<'a> {
    type Bar;
}

impl<'a, T> Foo<'a> for T {
    type Bar = &'a T; //~ ERROR E0309
}


fn main() { }
