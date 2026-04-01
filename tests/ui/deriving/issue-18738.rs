//@ check-pass
#![allow(dead_code)]
#[derive(Eq, PartialEq, PartialOrd, Ord)]
enum Test<'a> {
    Int(&'a isize),
    Slice(&'a [u8]),
}

#[derive(Eq, PartialEq, PartialOrd, Ord)]
struct Version {
    vendor_info: &'static str
}

#[derive(Eq, PartialEq, PartialOrd, Ord)]
struct Foo(&'static str);

fn main() {}
