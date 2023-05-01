trait Foo {
    fn assoc();
}

impl Foo for () {
    #[refine]
    //~^ ERROR the `#[refine]` attribute is an experimental feature
    fn assoc() {}
}

fn main() {}
