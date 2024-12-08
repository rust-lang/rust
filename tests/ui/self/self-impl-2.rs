//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
// Test that we can use `Self` types in impls in the expected way.


struct Foo;

// Test uses on inherent impl.
impl Foo {
    fn foo(_x: Self, _y: &Self, _z: Box<Self>) -> Self {
        Foo
    }

    fn baz() {
        // Test that Self cannot be shadowed.
        type Foo = i32;
        // There is no empty method on i32.
        Self::empty();

        let _: Self = Foo;
    }

    fn empty() {}
}

// Test uses when implementing a trait and with a type parameter.
pub struct Baz<X> {
    pub f: X,
}

trait SuperBar {
    type SuperQux;
}

trait Bar<X>: SuperBar {
    type Qux;

    fn bar(x: Self, y: &Self, z: Box<Self>, _: Self::SuperQux) -> Self;
    fn dummy(&self, x: X) { }
}

impl SuperBar for Box<Baz<isize>> {
    type SuperQux = bool;
}

impl Bar<isize> for Box<Baz<isize>> {
    type Qux = i32;

    fn bar(_x: Self, _y: &Self, _z: Box<Self>, _: Self::SuperQux) -> Self {
        let _: Self::Qux = 42;
        let _: <Self as Bar<isize>>::Qux = 42;

        let _: Self::SuperQux = true;
        let _: <Self as SuperBar>::SuperQux = true;

        Box::new(Baz { f: 42 })
    }
}

fn main() {
    let _: Foo = Foo::foo(Foo, &Foo, Box::new(Foo));
    let _: Box<Baz<isize>> = Bar::bar(Box::new(Baz { f: 42 }),
                                      &Box::new(Baz { f: 42 }),
                                      Box::new(Box::new(Baz { f: 42 })),
                                      true);
}
