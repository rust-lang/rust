//@ run-pass
// Regression test for issue #39292. The object vtable was being
// incorrectly left with a null pointer.

trait Foo<T> {
    fn print<'a>(&'a self) where T: 'a { println!("foo"); }
}

impl<'a> Foo<&'a ()> for () { }

trait Bar: for<'a> Foo<&'a ()> { }

impl Bar for () {}

fn main() {
    (&() as &dyn Bar).print(); // Segfault
}
