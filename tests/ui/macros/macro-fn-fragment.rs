//@ edition:2024

#![crate_type = "lib"]
#![feature(macro_attr)]
#![allow(incomplete_features)]
#![feature(macro_fragments_more)]

macro_rules! parse_fn {
    ($f:fn) => {};
}

parse_fn! {
    fn f1();
}

parse_fn! {
    pub async fn f2() {
        loop {}
    }
}

//~vv ERROR: expected a function
parse_fn! {
    struct S;
}

//~vv ERROR: expected identifier
parse_fn! {
    extern "C" fn;
}

macro_rules! fnattr {
    attr() ($f:fn) => { parse_fn!($f); };
}

#[fnattr]
fn f3() {}

#[fnattr]
enum E1 {
    V1,
}
//~^ ERROR: expected a function
