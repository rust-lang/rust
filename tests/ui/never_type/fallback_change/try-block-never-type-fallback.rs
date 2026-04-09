//@ check-pass
//@ edition: 2018..2024

// Issue #125364: Bad interaction between never_type, try_blocks, and From/Into
//
// Previouslyu, the never type in try blocks used to fall back to (),
// causing a type error (since (): Into<!> does not hold).
// Nowdays, it falls back to !, allowing the code to compile correctly.

#![feature(try_blocks)]

fn bar(_: Result<impl Into<!>, u32>) {
    unimplemented!()
}

fn foo(x: Result<!, u32>) {
    bar(try { x? });
}

fn main() {}
