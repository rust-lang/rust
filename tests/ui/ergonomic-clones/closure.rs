//@ check-pass
//@ edition:2018

#![feature(ergonomic_clones)]

fn ergonomic_clone_closure() -> i32 {
    let cl = use || {
        1
    };
    cl()
}

fn ergonomic_clone_async_closures() -> String {
    let s = String::from("hi");

    async use {
        22
    };

    s
}

fn main() {}
