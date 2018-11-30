// compile-flags: -Z chalk

trait Foo { }
trait Bar: Foo { }

impl Foo for i32 { }
impl Bar for i32 { }

fn only_foo<T: Foo>() { }

fn only_bar<T: Bar>() {
    // `T` implements `Bar` hence `T` must also implement `Foo`
    only_foo::<T>()
}

fn main() {
    only_bar::<i32>()
}
