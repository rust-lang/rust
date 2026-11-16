//@ compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/rust/issues/155035>

trait Foo {
    type V;
}

trait Callback<T: Foo>: Fn(T::V) {}
struct Bar<T: Foo> {
    callback: Box<dyn Callback<T>>,
}
impl Bar {
    //~^ ERROR: missing generics
    fn event(self) {
        (self.callback)(loop {})
    }
}

fn main() {}
