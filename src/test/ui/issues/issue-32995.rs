#![allow(unused)]

fn main() {
    let x: usize() = 1;
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait
    //~| WARN previously accepted
    //~| WARN hard error

    let b: ::std::boxed()::Box<_> = Box::new(1);
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait
    //~| WARN previously accepted
    //~| WARN hard error

    let p = ::std::str::()::from_utf8(b"foo").unwrap();
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait
    //~| WARN previously accepted
    //~| WARN hard error

    let p = ::std::str::from_utf8::()(b"foo").unwrap();
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait
    //~| WARN previously accepted
    //~| WARN hard error

    let o : Box<::std::marker()::Send> = Box::new(1);
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait
    //~| WARN previously accepted
    //~| WARN hard error

    let o : Box<Send + ::std::marker()::Sync> = Box::new(1);
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait
    //~| WARN previously accepted
    //~| WARN hard error
}

fn foo<X:Default>() {
    let d : X() = Default::default();
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait
    //~| WARN previously accepted
    //~| WARN hard error
}
