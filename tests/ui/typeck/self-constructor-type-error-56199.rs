// https://github.com/rust-lang/rust/issues/56199
enum Foo {}
struct Bar {}

impl Foo {
    fn foo() {
        let _ = Self;
        //~^ ERROR the `Self` constructor can only be used with tuple or unit structs
        let _ = Self();
        //~^ ERROR the `Self` constructor can only be used with tuple or unit structs
    }
}

impl Bar {
    fn bar() {
        let _ = Self;
        //~^ ERROR the `Self` constructor can only be used with tuple or unit structs
        let _ = Self();
        //~^ ERROR the `Self` constructor can only be used with tuple or unit structs
    }
}

fn main() {}
