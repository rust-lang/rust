struct Foo;

impl Foo {
    #[stable(feature = "rust1", since = "1.0.0")]
    //~^ ERROR expected item after attributes
}

fn main() {}
