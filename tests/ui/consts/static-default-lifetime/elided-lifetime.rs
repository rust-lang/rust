struct Foo<'a>(&'a ());

impl Foo<'_> {
    const STATIC: &str = "";
    //~^ ERROR missing lifetime specifier
}

trait Bar {
    const STATIC: &str;
}

impl Bar for Foo<'_> {
    const STATIC: &str = "";
    //~^ ERROR missing lifetime specifier
}

fn main() {}
