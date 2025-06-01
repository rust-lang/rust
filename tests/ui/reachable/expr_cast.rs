//@ edition: 2024
#![deny(unreachable_code)]

fn a() {
    _ = {return} as u32; //~ error: unreachable
}

fn b() {
    (return) as u32; //~ error: unreachable
}

fn main() {}
