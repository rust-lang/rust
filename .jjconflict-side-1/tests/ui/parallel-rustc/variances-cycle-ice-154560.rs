// Regression test for ICE from issue #154560.
//~^ ERROR cycle detected when computing the variances for items in this crate

//@ ignore-parallel-frontend query cycle + ICE

pub struct T<'a>(&'a str);

pub fn f<T>() -> _ {
    T
}

pub fn g<'a>(val: T<'a>) -> _ {
    T
}

fn main() {}
