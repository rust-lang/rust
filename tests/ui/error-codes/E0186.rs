trait Foo {
    fn foo(&self); //~ NOTE `&self` used in trait
}

struct Bar;

impl Foo for Bar {
    fn foo() {} //~ ERROR E0186
    //~^ NOTE expected `&self` in impl
}

fn main() {
}
