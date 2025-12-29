#![allow(incomplete_features)]
#![feature(fn_delegation)]

mod empty_glob {
    trait T {}

    struct S;

    reuse impl T for S { self.0 }
    //~^ ERROR empty glob delegation is not supported
}


fn main() {}
