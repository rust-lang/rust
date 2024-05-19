trait Foo {
    fn foo();
    //~^ NOTE trait method declared without `&self`
}

struct Bar;

impl Foo for Bar {
    fn foo(&self) {}
    //~^ ERROR E0185
    //~| NOTE `&self` used in impl
}

fn main() {
}
