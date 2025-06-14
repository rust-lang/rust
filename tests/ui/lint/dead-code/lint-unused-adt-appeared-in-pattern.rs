#![deny(dead_code)]

struct Foo(u8); //~ ERROR struct `Foo` is never constructed

enum Bar { //~ ERROR enum `Bar` is never used
    Var1(u8),
    Var2(u8),
}

pub trait Tr1 {
    fn f1() -> Self;
}

impl Tr1 for Foo {
    fn f1() -> Foo {
        let f = Foo(0);
        let Foo(tag) = f;
        Foo(tag)
    }
}

impl Tr1 for Bar {
    fn f1() -> Bar {
        let b = Bar::Var1(0);
        let b = if let Bar::Var1(_) = b {
            Bar::Var1(0)
        } else {
            Bar::Var2(0)
        };
        match b {
            Bar::Var1(_) => Bar::Var2(0),
            Bar::Var2(_) => Bar::Var1(0),
        }
    }
}

fn main() {}
