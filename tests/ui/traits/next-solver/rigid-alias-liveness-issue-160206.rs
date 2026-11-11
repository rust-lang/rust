//@ compile-flags: -Znext-solver=globally
//@ check-pass

trait Foo<'x> {
    type Out;
    fn foo(self) -> Self::Out;
}

struct Bar;

impl<'x> Foo<'x> for Bar {
    type Out = ();

    fn foo(self) -> Self::Out {
        todo!()
    }
}

fn make_static_foo<'x>(_: &'x ()) -> impl Foo<'x, Out: 'static> {
    Bar
}

fn test() {
    make_static_foo(&());
}

fn main() {}
