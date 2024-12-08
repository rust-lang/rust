struct Foo<A> { inner: A }

trait Bar { fn bar(); }

impl Bar for Foo<i32> {
    fn bar() {
        Self { inner: 1.5f32 }; //~ ERROR mismatched types
    }
}

impl<T> Foo<T> {
    fn new<U>(u: U) -> Foo<U> {
        Self {
        //~^ ERROR mismatched types
            inner: u
            //~^ ERROR mismatched types
        }
    }
}

fn main() {}
