struct Foo<'a>(&'a ());

fn test(_: Foo) {}

#[deny(elided_lifetimes_in_paths)]
mod w {
    fn test2(_: super::Foo) {}
    //~^ ERROR hidden lifetime parameters in types are deprecated
}

fn main() {}
