// check-pass

fn any<T>() -> T {
    loop {}
}

trait Foo {
    type V;
}

trait Callback<T: Foo>: Fn(&T, &T::V) {}
impl<T: Foo, F: Fn(&T, &T::V)> Callback<T> for F {}

struct Bar<T: Foo> {
    callback: Box<dyn Callback<T>>,
}

impl<T: Foo> Bar<T> {
    fn event(&self) {
        (self.callback)(any(), any());
    }
}

struct A;
struct B;
impl Foo for A {
    type V = B;
}

fn main() {
    let foo = Bar::<A> { callback: Box::new(|_: &A, _: &B| ()) };
    foo.event();
}
