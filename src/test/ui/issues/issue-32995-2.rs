#![allow(unused)]

fn main() {
    { fn f<X: ::std::marker()::Send>() {} }
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait
    //~| WARN previously accepted
    //~| WARN hard error

    { fn f() -> impl ::std::marker()::Send { } }
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait
    //~| WARN previously accepted
    //~| WARN hard error
}

#[derive(Clone)]
struct X;

impl ::std::marker()::Copy for X {}
//~^ ERROR parenthesized type parameters may only be used with a `Fn` trait
//~| WARN previously accepted
//~| WARN hard error
