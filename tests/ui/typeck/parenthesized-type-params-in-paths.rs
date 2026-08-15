//! Regression test for <https://github.com/rust-lang/rust/issues/32995>.
//! Test parenthesized type params syntax is forbidden for non `Fn` trait,
//! and can't be used in paths.

fn main() {
    let x: usize() = 1;
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait

    let b: ::std::boxed()::Box<_> = Box::new(1);
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait

    let p = ::std::str::()::from_utf8(b"foo").unwrap();
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait

    let p = ::std::str::from_utf8::()(b"foo").unwrap();
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait

    let o : Box<dyn (::std::marker()::Send)> = Box::new(1);
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait

    let o : Box<dyn Send + ::std::marker()::Sync> = Box::new(1);
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait

    { fn f<X: ::std::marker()::Send>() {} }
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait

    { fn f() -> impl ::std::marker()::Send { } }
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait

}

#[derive(Clone)]
struct X;

impl ::std::marker()::Copy for X {}
//~^ ERROR parenthesized type parameters may only be used with a `Fn` trait

fn foo<X:Default>() {
    let d : X() = Default::default();
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait
}
