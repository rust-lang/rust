// run-pass
// All 3 expressions should work in that the argument gets
// coerced to a trait object

// pretty-expanded FIXME #23616

fn main() {
    send::<Box<dyn Foo>>(Box::new(Output(0)));
    Test::<Box<dyn Foo>>::foo(Box::new(Output(0)));
    Test::<Box<dyn Foo>>::new().send(Box::new(Output(0)));
}

fn send<T>(_: T) {}

struct Test<T> { marker: std::marker::PhantomData<T> }
impl<T> Test<T> {
    fn new() -> Test<T> { Test { marker: ::std::marker::PhantomData } }
    fn foo(_: T) {}
    fn send(&self, _: T) {}
}

trait Foo { fn dummy(&self) { }}
struct Output(isize);
impl Foo for Output {}
