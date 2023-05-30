// check-pass

#![allow(unconditional_recursion)]

fn test<'a>() -> impl Sized + 'a {
    let r: i32 = test();
    r
}

fn main() {}
