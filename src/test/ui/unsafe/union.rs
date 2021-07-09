// run-pass
// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![feature(untagged_unions)]

union Foo {
    bar: i8,
    zst: (),
    pizza: Pizza,
}

struct Pizza {
    topping: Option<PizzaTopping>
}

#[allow(dead_code)]
enum PizzaTopping {
    Cheese,
    Pineapple,
}

struct FooHolder {
    inner_foo: Foo
}

fn do_nothing(_x: &mut Foo) {}

pub fn main() {
    let mut foo = Foo { bar: 5 };
    do_nothing(&mut foo);
    foo.bar = 6;
    unsafe { foo.bar += 1; }
    assert_eq!(unsafe { foo.bar }, 7);
    unsafe {
        let Foo { bar: inner } = foo;
        assert_eq!(inner, 7);
    }
    let foo = if let true = true { foo } else { foo };

    unsafe {
        match foo {
            Foo { bar: _a } => {},
        }
    }
    unsafe {
        match foo {
            Foo {
                pizza: Pizza {
                    topping: Some(PizzaTopping::Cheese) | Some(PizzaTopping::Pineapple) | None
                }
            } => {},
        }
    }
    // binding to wildcard is okay
    match foo {
        Foo { bar: _ } => {},
    }
    let Foo { bar: _ } = foo;
    // MIR unsafeck incorrectly thinks that it is safe to do these
    unsafe { //[mir]~ WARNING
        match foo {
            Foo { zst: () } => {},
        }
    }
    unsafe { //[mir]~ WARNING
        match foo {
            Foo { pizza: Pizza { .. } } => {},
        }
    }
    let foo = Foo { bar: 5 };
    let foo = if let 3 = if let true = true { 3 } else { 4 } { foo } else { foo };

    let (_foo2, _random) = (foo, 42);

    let mut foo_holder = FooHolder { inner_foo: Foo { bar: 5 } };
    foo_holder.inner_foo.bar = 4;
    assert_eq!(unsafe { foo_holder.inner_foo.bar }, 4);
    drop(foo_holder);
}
