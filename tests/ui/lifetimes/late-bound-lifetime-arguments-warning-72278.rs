// https://github.com/rust-lang/rust/issues/72278
// and https://github.com/rust-lang/rust/issues/42868
//@ run-pass

#![allow(unused)]

struct S;

impl S {
    fn func<'a, U>(&'a self) -> U {
        todo!()
    }
}

fn dont_crash<'a, U>() -> U {
    S.func::<'a, U>()
    //~^ WARN cannot specify lifetime arguments explicitly
    //~| WARN this was previously accepted
}

fn main() {}
