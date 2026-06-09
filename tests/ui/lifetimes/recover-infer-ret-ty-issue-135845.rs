// Regression test for #135845.

use std::marker::PhantomData;

fn b<'a>() -> _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types [E0121]
    let _: PhantomData<&'a ()> = PhantomData;
    0
}

fn main() {}
