trait Foo {
    const NAME: &'static str;
}


impl<'a> Foo for &'a () {
    const NAME: &'a str = "unit";
    //~^ ERROR const not compatible with trait
}

fn main() {}
