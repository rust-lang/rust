//@ edition: 2021
#![allow(incomplete_features)]
#![feature(mut_ref)]

struct Foo(u8);

fn main() {
    let Foo(a) = Foo(0);
    a = 42; //~ ERROR [E0384]

    let Foo(mut a) = Foo(0);
    a = 42;

    let Foo(ref a) = Foo(0);
    a = &42; //~ ERROR [E0384]

    let Foo(mut ref a) = Foo(0);
    a = &42;

    let Foo(ref mut a) = Foo(0);
    a = &mut 42; //~ ERROR [E0384]

    let Foo(mut ref mut a) = Foo(0);
    a = &mut 42;

    let Foo(a) = &Foo(0);
    a = &42; //~ ERROR [E0384]

    let Foo(mut a) = &Foo(0);
    a = 42;

    let Foo(ref a) = &Foo(0);
    a = &42; //~ ERROR [E0384]

    let Foo(mut ref a) = &Foo(0);
    a = &42;

    let Foo(a) = &mut Foo(0);
    a = &mut 42; //~ ERROR [E0384]

    let Foo(mut a) = &mut Foo(0);
    a = 42;

    let Foo(ref a) = &mut Foo(0);
    a = &42; //~ ERROR [E0384]

    let Foo(mut ref a) = &mut Foo(0);
    a = &42;

    let Foo(ref mut a) = &mut Foo(0);
    a = &mut 42; //~ ERROR [E0384]

    let Foo(mut ref mut a) = &mut Foo(0);
    a = &mut 42;
}
