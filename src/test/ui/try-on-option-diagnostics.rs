#![feature(try_trait)]
// edition:2018
fn main() {}

fn a_function() -> u32 {
    let x: Option<u32> = None;
    x?; //~ ERROR the `?` operator
    22
}

fn a_closure() -> u32 {
    let a_closure = || {
        let x: Option<u32> = None;
        x?; //~ ERROR the `?` operator
        22
    };
    a_closure()
}
