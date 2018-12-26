struct Foo;

impl Foo {
    fn bar(&self) {}
}

trait MyTrait {
    fn trait_bar() {}
}

impl MyTrait for Foo {}

fn main() {
    match 0u32 {
        Foo::bar => {}
        //~^ ERROR expected unit struct/variant or constant, found method `<Foo>::bar`
    }
    match 0u32 {
        <Foo>::bar => {}
        //~^ ERROR expected unit struct/variant or constant, found method `<Foo>::bar`
    }
    match 0u32 {
        <Foo>::trait_bar => {}
        //~^ ERROR expected unit struct/variant or constant, found method `<Foo>::trait_bar`
    }
}
