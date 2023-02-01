trait Foo {
    fn bar(&self, x: usize) -> Self;
}
impl Foo for isize {
    fn bar(&self) -> isize {
        //~^ ERROR method `bar` has 1 parameter but the declaration in trait `Foo::bar` has 2
        //~| HELP: modify the signature to match the trait definition
        *self
    }
}

fn main() {
}
