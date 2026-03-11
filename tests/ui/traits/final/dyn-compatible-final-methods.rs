//@ run-pass

#![feature(final_associated_functions)]

trait Tr
where
    Self: 'static,
{
    final fn foo(&self) -> std::any::TypeId {
        std::any::TypeId::of::<Self>()
    }
}

struct Foo;
impl Tr for Foo {}

struct Bar;
impl Tr for Bar {}

fn foo(t: &dyn Tr) -> std::any::TypeId {
    t.foo()
}

fn main() {
    assert_ne!(foo(&Foo), foo(&Bar));
}
