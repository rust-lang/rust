//@ run-pass

trait Foo<'a> {
    fn xyz(self);
}
impl<'a, T> Foo<'a> for T where 'static: 'a {
    fn xyz(self) {}
}

trait Bar {
    fn uvw(self);
}
impl<T> Bar for T where for<'a> T: Foo<'a> {
    fn uvw(self) { self.xyz(); }
}

fn foo<T>(t: T) where T: Bar {
    t.uvw();
}

fn main() {
    foo(0);
}
