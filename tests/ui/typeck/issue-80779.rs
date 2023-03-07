// Regression test for #80779.

pub struct T<'a>(&'a str);

pub fn f<'a>(val: T<'a>) -> _ {
    //~^ ERROR: the placeholder `_` is not allowed within types on item signatures for return types
    g(val)
}

pub fn g(_: T<'static>) -> _ {}
//~^ ERROR: the placeholder `_` is not allowed within types on item signatures for return types

fn main() {}
