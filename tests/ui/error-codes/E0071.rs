enum Foo {}
type FooAlias = Foo;

fn main() {
    let u = FooAlias { value: 0 };
    //~^ ERROR expected struct, variant or union type, found `Foo` [E0071]
}
