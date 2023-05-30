#![allow(unconditional_recursion)]

fn foo<'a: 'a>() -> impl Sized {
    let _: *mut &'a () = foo::<'a>();
    //~^ ERROR
    loop {}
}

fn main() {}
