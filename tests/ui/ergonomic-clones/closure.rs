//@ check-pass
//@ edition:2018

#![feature(ergonomic_clones)]

fn ergonomic_clone_closure_no_captures() -> i32 {
    let cl = use || {
        1
    };
    cl()
}

fn ergonomic_clone_closure_with_captures() -> String {
    let s = String::from("hi");

    let cl = use || {
        s
    };
    cl()
}

fn ergonomic_clone_async_closures() -> String {
    let s = String::from("hi");

    async use {
        s
    };

    s
}

fn main() {}
