// Test for issue #67776: binding named the same as enum variant
// should report an error even when matching against a reference type

#![allow(unused_variables)]
#![allow(non_snake_case)]

enum Foo {
    Bar,
    Baz,
}


fn fn1(e: Foo) {
    match e {
        Bar => {},
        //~^ ERROR named the same as one of the variants of the type `Foo`
        Baz => {},
        //~^ ERROR named the same as one of the variants of the type `Foo`
    }
}

fn fn2(e: &Foo) {
    match e {
        Bar => {},
        //~^ ERROR named the same as one of the variants of the type `Foo`
        Baz => {},
        //~^ ERROR named the same as one of the variants of the type `Foo`
    }
}

fn fn3(e: &mut &&mut Foo) {
    match e {
        Bar => {},
        //~^ ERROR named the same as one of the variants of the type `Foo`
        Baz => {},
        //~^ ERROR named the same as one of the variants of the type `Foo`
    }
}

fn main() {}
