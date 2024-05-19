trait Trait {
    type Bound<'a>;
}

fn foo() where Trait<for<'a> Bound<'a> = &'a ()> {
    //~^ ERROR `for<...>` is not allowed on associated type bounds
}

fn main() {}
