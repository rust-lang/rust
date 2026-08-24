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
    //~^ ERROR cannot find value `A` in this scope
    B.foo();
    //~^ ERROR cannot find value `B` in this scope
    C.foo();
    //~^ ERROR cannot find value `C` in this scope
    D.foo();
    //~^ ERROR cannot find value `D` in this scope
    E.foo();
    //~^ ERROR cannot find value `E` in this scope

    // Only tuple variants are suggested in calls or tuple struct pattern matching.

    let x = A(3);
    //~^ ERROR cannot find function, tuple struct or tuple variant `A` in this scope
    if let A(3) = x { }
    //~^ ERROR cannot find tuple struct or tuple variant `A` in this scope
}
