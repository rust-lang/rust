trait Foo {
    fn a();
}

struct Bar;

impl Foo for Bar {
    fn a() {}
    fn b() {}
    //~^ ERROR E0407
}

fn main() {
}
