#![feature(default_field_values)]
struct A<'a> {
    x: Vec<A> = Vec::new(), //~ ERROR missing lifetime specifier
}

fn main() {}
