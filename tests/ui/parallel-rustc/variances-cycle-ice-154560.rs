// Regression test for ICE from issue #154560.

pub struct T<'a>(&'a str);

pub fn f<T>() -> _ { //~ ERROR placeholder `_` is not allowed
    T
}

pub fn g<'a>(val: T<'a>) -> _ { //~ ERROR placeholder `_` is not allowed
    T
}

fn main() {}
