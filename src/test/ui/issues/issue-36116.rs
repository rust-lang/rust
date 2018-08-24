// Unnecessary path disambiguator is ok

#![feature(rustc_attrs)]
#![allow(unused)]

macro_rules! m {
    ($p: path) => {
        let _ = $p(0);
        let _: $p;
    }
}

struct Foo<T> {
    _a: T,
}

struct S<T>(T);

fn f() {
    let f = Some(Foo { _a: 42 }).map(|a| a as Foo::<i32>); //~ WARN unnecessary path disambiguator
    let g: Foo::<i32> = Foo { _a: 42 }; //~ WARN unnecessary path disambiguator

    m!(S::<u8>); // OK, no warning
}

#[rustc_error]
fn main() {} //~ ERROR compilation successful
