//@ check-pass

#[repr(Rust)]
struct A;

#[repr(Rust, align(16))]
struct B;

#[repr(Rust, packed)]
struct C;

fn main() {}
