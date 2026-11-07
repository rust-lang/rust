//@ check-pass

#![feature(cfg_accessible)]

mod private {
    struct Struct;
    enum Enum{}
    union Union{_a:u8}
}

#[cfg_accessible(private::Struct)]
const A: bool = true;

#[cfg_accessible(private::Enum)]
const A: bool = true;

#[cfg_accessible(private::Union)]
const A: bool = true;

const A: bool = false; // Will conflict if any of those is accessible
fn main() {}
