enum Foo {
    //~^ HELP consider importing this tuple variant
    A(u32),
    B(u32),
}

enum Bar {
    C(u32),
    D(u32),
    E,
    F,
}

fn main() {
    let _: Foo = Foo(0);
    //~^ ERROR expected function
    //~| HELP try to construct one of the enum's variants

    let _: Foo = Foo.A(0);
    //~^ ERROR expected value, found enum `Foo`
    //~| HELP use the path separator to refer to a variant

    let _: Foo = Foo.Bad(0);
    //~^ ERROR expected value, found enum `Foo`
    //~| HELP the following enum variants are available

    let _: Bar = Bar(0);
    //~^ ERROR expected function
    //~| HELP try to construct one of the enum's variants
    //~| HELP you might have meant to construct one of the enum's non-tuple variants

    let _: Bar = Bar.C(0);
    //~^ ERROR expected value, found enum `Bar`
    //~| HELP use the path separator to refer to a variant

    let _: Bar = Bar.E;
    //~^ ERROR expected value, found enum `Bar`
    //~| HELP use the path separator to refer to a variant

    let _: Bar = Bar.Bad(0);
    //~^ ERROR expected value, found enum `Bar`
    //~| HELP you might have meant to use one of the following enum variants
    //~| HELP alternatively, the following enum variants are also available

    let _: Bar = Bar.Bad;
    //~^ ERROR expected value, found enum `Bar`
    //~| HELP you might have meant to use one of the following enum variants
    //~| HELP alternatively, the following enum variants are also available

    match Foo::A(42) {
        A(..) => {}
        //~^ ERROR cannot find tuple struct or tuple variant `A` in this scope
        Foo(..) => {}
        //~^ ERROR expected tuple struct or tuple variant
        //~| HELP try to match against one of the enum's variants
        _ => {}
    }
}
