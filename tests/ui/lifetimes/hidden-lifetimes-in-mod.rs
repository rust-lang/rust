struct Foo<'a>(&'a ());

fn test(_: Foo) {}

#[deny(hidden_lifetimes_in_paths)]
mod w {
    fn test2(_: super::Foo) {}
    //~^ ERROR paths containing hidden lifetime parameters are deprecated
}

fn main() {}
