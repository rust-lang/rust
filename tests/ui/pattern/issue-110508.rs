//@ run-pass

#![deny(dead_code)]

#[derive(PartialEq, Eq)]
pub enum Foo {
    FooA(()),
    FooB(Vec<()>),
}

impl Foo {
    const A1: Foo = Foo::FooA(());
    const A2: Foo = Self::FooA(());
    const A3: Self = Foo::FooA(());
    const A4: Self = Self::FooA(());
    const A5: u32 = 1;
}

fn main() {
    let foo = Foo::FooA(());

    match foo {
        Foo::A1 => {},
        _ => {},
    }

    match foo {
        Foo::A2 => {},
        _ => {},
    }

    match foo {
        Foo::A3 => {},
        _ => {},
    }

    match foo {
        Foo::A4 => {},
        _ => {},
    }

    match 3 {
        Foo::A5..5 => {}
        _ => {}
    }
}
