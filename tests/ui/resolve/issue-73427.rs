enum A {
    StructWithFields { x: () },
    TupleWithFields(()),
    Struct {},
    Tuple(),
    Unit,
}

enum B {
    StructWithFields { x: () },
    TupleWithFields(()),
}

enum C {
    StructWithFields { x: () },
    TupleWithFields(()),
    Unit,
}

enum D {
    TupleWithFields(()),
    Unit,
}

enum E {
    TupleWithFields(()),
}

fn main() {
    // Only variants without fields are suggested (and others mentioned in a note) where an enum
    // is used rather than a variant.

    A.foo();
    //~^ ERROR expected value, found enum `A`
    B.foo();
    //~^ ERROR expected value, found enum `B`
    C.foo();
    //~^ ERROR expected value, found enum `C`
    D.foo();
    //~^ ERROR expected value, found enum `D`
    E.foo();
    //~^ ERROR expected value, found enum `E`

    // Only tuple variants are suggested in calls or tuple struct pattern matching.

    let x = A(3);
    //~^ ERROR expected function, tuple struct or tuple variant, found enum `A`
    if let A(3) = x { }
    //~^ ERROR expected tuple struct or tuple variant, found enum `A`
}
