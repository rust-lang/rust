struct Foo;

trait MyTrait {
    fn trait_bar() {}
}

impl MyTrait for Foo {}

fn main() {
    match 0u32 {
        <Foo as MyTrait>::trait_bar => {}
        //~^ ERROR expected unit struct, unit variant or constant, found associated function
    }
}
