// Struct literal expressions.

fn main() {
    let x = Bar;

    // Comment
    let y = Foo { a: x };

    Foo { a: Bar, b: foo() };

    Foo { a: foo(), b: bar(), ..something };

    Foooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo { a: foo(), b: bar() };
    Fooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo { a: foo(),
                                                                               b: bar(), };

    Fooooooooooooooooooooooooooooooooooooooooooooooooooooo { a: foo(),
                                                             b: bar(),
                                                             c: bar(),
                                                             d: bar(),
                                                             e: bar(),
                                                             f: bar(),
                                                             ..baz() };
}
