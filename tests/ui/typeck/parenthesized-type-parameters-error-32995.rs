// https://github.com/rust-lang/rust/issues/32995
fn main() {
    { fn f<X: ::std::marker()::Send>() {} }
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait

    { fn f() -> impl ::std::marker()::Send { } }
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait
}

#[derive(Clone)]
struct X;

impl ::std::marker()::Copy for X {}
//~^ ERROR parenthesized type parameters may only be used with a `Fn` trait
