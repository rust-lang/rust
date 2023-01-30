// run-pass
trait Foo: Sized {
    fn foo(self) {}
}

trait Bar: Sized {
    fn bar(self) {}
}

struct S;

impl<'l> Foo for &'l S {}

impl<T: Foo> Bar for T {}

fn main() {
    let s = S;
    s.foo();
    (&s).bar();
    s.bar();
}
