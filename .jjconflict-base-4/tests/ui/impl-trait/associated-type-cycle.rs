trait Foo {
    type Bar;
    fn foo(self) -> Self::Bar;
}

impl Foo for Box<dyn Foo> {
    //~^ ERROR: the value of the associated type `Bar` in `Foo` must be specified
    type Bar = <Self as Foo>::Bar;
    fn foo(self) -> <Self as Foo>::Bar {
        (*self).foo()
    }
}

fn main() {}
