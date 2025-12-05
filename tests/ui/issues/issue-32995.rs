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
}

fn foo<X:Default>() {
    let d : X() = Default::default();
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait
}
