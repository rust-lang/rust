// compile-flags: -Z chalk

trait Foo { }

trait Bar {
    type Item: Foo;
}

impl Foo for i32 { }
impl Bar for i32 {
    type Item = i32;
}

fn only_foo<T: Foo>() { }

fn only_bar<T: Bar>() {
    // `T` implements `Bar` hence `<T as Bar>::Item` must also implement `Bar`
    only_foo::<T::Item>()
}

fn main() {
    only_bar::<i32>();
    only_foo::<<i32 as Bar>::Item>();
}
