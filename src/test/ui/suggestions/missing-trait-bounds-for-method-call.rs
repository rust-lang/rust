#[derive(Default, PartialEq)]
struct Foo<T> {
    bar: Box<[T]>,
}

trait Bar {
    fn foo(&self) {}
}

impl<T: Default + Bar> Bar for Foo<T> {}

impl<T> Foo<T> {
    fn bar(&self) {
        self.foo();
        //~^ ERROR no method named `foo` found for reference `&Foo<T>` in the current scope
    }
}

struct Fin<T> where T: Bar {
    bar: Box<[T]>,
}

impl<T: Default + Bar> Bar for Fin<T> {}

impl<T: Bar> Fin<T> {
    fn bar(&self) {
        self.foo();
        //~^ ERROR no method named `foo` found for reference `&Fin<T>` in the current scope
    }
}
fn main() {}
