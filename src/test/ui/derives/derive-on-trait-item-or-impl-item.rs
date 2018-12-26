trait Foo {
    #[derive(Clone)]
    //~^ ERROR `derive` may only be applied to structs, enums and unions
    type Bar;
}

impl Bar {
    #[derive(Clone)]
    //~^ ERROR `derive` may only be applied to structs, enums and unions
    fn bar(&self) {}
}

fn main() {}
