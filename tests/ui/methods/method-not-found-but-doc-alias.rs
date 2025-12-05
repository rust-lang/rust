struct Foo;

impl Foo {
    #[doc(alias = "quux")]
    fn bar(&self) {}
}

fn main() {
    Foo.quux();
    //~^ ERROR  no method named `quux` found for struct `Foo` in the current scope
}
