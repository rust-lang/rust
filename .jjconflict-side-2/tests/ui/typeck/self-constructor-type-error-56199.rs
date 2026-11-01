// https://github.com/rust-lang/rust/issues/56199
enum Foo {}
enum Lab {
    Qux,
}
struct Bar {}

impl Foo {
    fn foo() {
        let _ = Self;
        //~^ ERROR the `Self` constructor can only be used with tuple or unit structs
        let _ = Self();
        //~^ ERROR the `Self` constructor can only be used with tuple or unit structs
    }
    fn foo_method(self) {
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
    fn bar_method(self) {
        let _ = Self;
        //~^ ERROR the `Self` constructor can only be used with tuple or unit structs
        let _ = Self();
        //~^ ERROR the `Self` constructor can only be used with tuple or unit structs
    }
}

impl Lab {
    fn lab() {
        let _ = Self;
        //~^ ERROR the `Self` constructor can only be used with tuple or unit structs
        let _ = Self();
        //~^ ERROR the `Self` constructor can only be used with tuple or unit structs
    }
    fn lab_method(self) {
        let _ = Self;
        //~^ ERROR the `Self` constructor can only be used with tuple or unit structs
        let _ = Self();
        //~^ ERROR the `Self` constructor can only be used with tuple or unit structs
    }
}


fn main() {}
