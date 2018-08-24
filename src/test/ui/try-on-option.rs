#![feature(try_trait)]

fn main() {}

fn foo() -> Result<u32, ()> {
    let x: Option<u32> = None;
    x?; //~ the trait bound
    Ok(22)
}

fn bar() -> u32 {
    let x: Option<u32> = None;
    x?; //~ the `?` operator
    22
}
