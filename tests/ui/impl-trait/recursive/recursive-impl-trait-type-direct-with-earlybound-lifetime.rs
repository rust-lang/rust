// check-pass

#![allow(unconditional_recursion)]

fn foo<'a: 'a>() -> impl Sized {
    let _: () = foo::<'a>();
    loop {}
}

fn main() {}
