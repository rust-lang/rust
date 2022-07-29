trait Foo {
    const NAME: &'static str;
}


impl<'a> Foo for &'a () {
    const NAME: &'a str = "unit";
    //~^ ERROR mismatched types [E0308]
}

fn main() {}
