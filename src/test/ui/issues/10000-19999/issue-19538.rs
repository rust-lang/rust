trait Foo {
    fn foo<T>(&self, val: T);
}

trait Bar: Foo { }

pub struct Thing;

impl Foo for Thing {
    fn foo<T>(&self, val: T) { }
}

impl Bar for Thing { }

fn main() {
    let mut thing = Thing;
    let test: &mut dyn Bar = &mut thing;
    //~^ ERROR E0038
    //~| ERROR E0038
}
