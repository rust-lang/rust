//@ known-bug: #120600
#![feature(never_type)]
#![feature(never_type_fallback)]

#[derive(Ord, Eq, PartialOrd, PartialEq)]
enum E {
    Foo,
    Bar(!, i32, i32),
}

fn main() {}
