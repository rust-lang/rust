
enum Foo {}
struct Bar {}

impl Foo {
    fn foo() {
        let _ = Self;
        //~^ ERROR the `Self` constructor can only be used with tuple structs
        let _ = Self();
        //~^ ERROR the `Self` constructor can only be used with tuple structs
    }
}

impl Bar {
    fn bar() {
        let _ = Self;
        //~^ ERROR the `Self` constructor can only be used with tuple structs
        let _ = Self();
        //~^ ERROR the `Self` constructor can only be used with tuple structs
    }
}

fn main() {}
