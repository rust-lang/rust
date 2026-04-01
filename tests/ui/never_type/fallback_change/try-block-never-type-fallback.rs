//@ revisions: e2021 e2024
//@[e2021] edition: 2021
//@[e2024] edition: 2024
//@[e2024] check-pass

// Issue #125364: Bad interaction between never_type, try_blocks, and From/Into
//
// In edition 2021, the never type in try blocks falls back to (),
// causing a type error (since (): Into<!> does not hold).
// In edition 2024, it falls back to !, allowing the code to compile correctly.

#![feature(never_type)]
#![feature(try_blocks)]

fn bar(_: Result<impl Into<!>, u32>) {
    unimplemented!()
}

fn foo(x: Result<!, u32>) {
    bar(try { x? });
    //[e2021]~^ ERROR the trait bound `!: From<()>` is not satisfied
}

fn main() {
}
