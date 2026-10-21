#![feature(rustc_attrs)]
#![deny(dead_code)]

#[derive(Clone)]
struct Foo(usize); //~ ERROR: struct `Foo` is never constructed

#[rustc_trivial_field_reads]
fn main() {
    println!("{}", Foo(0).0);
}
