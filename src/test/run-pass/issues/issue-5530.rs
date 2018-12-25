// run-pass
#![allow(dead_code)]

enum Enum {
    Foo { foo: usize },
    Bar { bar: usize }
}

fn fun1(e1: &Enum, e2: &Enum) -> usize {
    match (e1, e2) {
        (&Enum::Foo { foo: _ }, &Enum::Foo { foo: _ }) => 0,
        (&Enum::Foo { foo: _ }, &Enum::Bar { bar: _ }) => 1,
        (&Enum::Bar { bar: _ }, &Enum::Bar { bar: _ }) => 2,
        (&Enum::Bar { bar: _ }, &Enum::Foo { foo: _ }) => 3,
    }
}

fn fun2(e1: &Enum, e2: &Enum) -> usize {
    match (e1, e2) {
        (&Enum::Foo { foo: _ }, &Enum::Foo { foo: _ }) => 0,
        (&Enum::Foo { foo: _ }, _              ) => 1,
        (&Enum::Bar { bar: _ }, &Enum::Bar { bar: _ }) => 2,
        (&Enum::Bar { bar: _ }, _              ) => 3,
    }
}

pub fn main() {
    let foo = Enum::Foo { foo: 1 };
    let bar = Enum::Bar { bar: 1 };

    assert_eq!(fun1(&foo, &foo), 0);
    assert_eq!(fun1(&foo, &bar), 1);
    assert_eq!(fun1(&bar, &bar), 2);
    assert_eq!(fun1(&bar, &foo), 3);

    assert_eq!(fun2(&foo, &foo), 0);
    assert_eq!(fun2(&foo, &bar), 1); // fun2 returns 0
    assert_eq!(fun2(&bar, &bar), 2);
    assert_eq!(fun2(&bar, &foo), 3); // fun2 returns 2
}
