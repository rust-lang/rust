#![allow(unused)]
enum Foo {
    Foo1(Box<u32>, Box<u32>),
    Foo2(Box<u32>),
    Foo3(Box<u32>),
    Foo4(Box<u32>),
    Foo5(Box<u32>),
    Foo6(Box<u32>),
    Foo7(Box<u32>),
    Foo8(Box<u32>),
}

fn blah() {
    let f = &Foo::Foo1(Box::new(1), Box::new(2));
    match *f { //~ ERROR cannot move out of
        Foo::Foo1(num1,
                  num2) => (),
        Foo::Foo2(num) => (),
        Foo::Foo3(num) => (),
        Foo::Foo4(num) => (),
        Foo::Foo5(num) => (),
        Foo::Foo6(num) => (),
        Foo::Foo7(num) => (),
        Foo::Foo8(num) => (),
    }
}

fn main() {}
