#![feature(type_alias_impl_trait)]
//@ check-pass
//@ revisions: default edition2021
//@[edition2021]edition: 2021

fn main() {
    type T = impl Copy;
    let foo: T = Some((1u32, 2u32));
    match foo {
        None => (),
        Some((a, b)) => (),
    }
}

fn upvar() {
    #[derive(Copy, Clone)]
    struct Foo((u32, u32));

    type T = impl Copy;
    let foo: T = Foo((1u32, 2u32));
    let x = move || {
        let Foo((a, b)) = foo;
    };
}

fn enum_upvar() {
    type T = impl Copy;
    let foo: T = Some((1u32, 2u32));
    let x = move || match foo {
        None => (),
        Some((a, b)) => (),
    };
}

fn r#struct() {
    #[derive(Copy, Clone)]
    struct Foo((u32, u32));

    type U = impl Copy;
    let foo: U = Foo((1u32, 2u32));
    let Foo((a, b)) = foo;
}

mod only_pattern {
    type T = impl Copy;

    #[define_opaque(T)]
    fn foo(foo: T) {
        let (mut x, mut y) = foo;
        x = 42;
        y = "foo";
    }

    type U = impl Copy;

    #[define_opaque(U)]
    fn bar(bar: Option<U>) {
        match bar {
            Some((mut x, mut y)) => {
                x = 42;
                y = "foo";
            }
            None => {}
        }
    }

    type V = impl Copy;

    #[define_opaque(V)]
    fn baz(baz: Option<V>) {
        match baz {
            _ => {}
            Some((mut x, mut y)) => {
                x = 42;
                y = "foo";
            }
            None => {}
        }
    }
}

mod only_pattern_rpit {
    #[allow(unconditional_recursion)]
    fn foo(b: bool) -> impl Copy {
        let (mut x, mut y) = foo(false);
        x = 42;
        y = "foo";
        if b { panic!() } else { foo(true) }
    }

    fn bar(b: bool) -> Option<impl Copy> {
        if b {
            return None;
        }
        match bar(!b) {
            Some((mut x, mut y)) => {
                x = 42;
                y = "foo";
            }
            None => {}
        }
        None
    }
}
