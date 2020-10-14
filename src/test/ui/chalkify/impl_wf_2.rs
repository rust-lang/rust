// Split out of impl_wf.rs to work around rust aborting compilation early

// compile-flags: -Z chalk

trait Foo: Sized { }

trait Bar {
    type Item: Foo;
}

impl Foo for i32 { }

// Implicit `T: Sized` bound.
impl<T> Foo for Option<T> { }

impl Bar for () {
    type Item = i32;
}

impl<T> Bar for Option<T> {
    type Item = Option<T>;
}

impl Bar for f32 {
    type Item = f32;
    //~^ ERROR the trait bound `f32: Foo` is not satisfied
}

trait Baz<U: ?Sized> where U: Foo { }

impl Baz<i32> for i32 { }

fn main() {}
