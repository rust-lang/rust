// check-pass

#![feature(type_alias_impl_trait)]
fn foo() {
    struct Foo<'a> {
        x: &'a mut u8,
    }
    impl<'a> Foo<'a> {
        fn foo(&self) -> impl Sized {}
    }
    // use site
    let mut x = 5;
    let y = Foo { x: &mut x };
    let z = y.foo();
    let _a = &x; // invalidate the `&'a mut`in `y`
    let _b = z; // this should *not* check that `'a` in the type `Foo<'a>::foo::opaque` is live
}

fn bar() {
    struct Foo<'a> {
        x: &'a mut u8,
    }

    // desugared
    type FooX = impl Sized;
    impl<'a> Foo<'a> {
        fn foo(&self) -> FooX {}
    }

    // use site
    let mut x = 5;
    let y = Foo { x: &mut x };
    let z = y.foo();
    let _a = &x; // invalidate the `&'a mut`in `y`
    let _b = z; // this should *not* check that `'a` in the type `Foo<'a>::foo::opaque` is live
}

fn main() {}
