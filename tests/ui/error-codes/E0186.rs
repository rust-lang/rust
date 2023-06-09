trait Foo {
    fn foo(&self); //~ `&self` used in trait
}

struct Bar;

impl Foo for Bar {
    fn foo() {} //~ ERROR E0186
    //~^ expected `&self` in impl
}

fn main() {
}
