// issue: #111906
//@ check-pass

#![allow(unconditional_recursion)]

fn foo<'a: 'a>() -> impl Sized {
    let _: () = foo::<'a>();
    loop {}
}

fn bar<'a: 'a>() -> impl Sized + 'a {
    let _: *mut &'a () = bar::<'a>();
    loop {}
}

fn main() {}
