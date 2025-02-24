#![feature(default_field_values)]
struct A<'a> { //~ ERROR lifetime parameter `'a` is never used
    x: Vec<A> = Vec::new(), //~ ERROR missing lifetime specifier
}

fn main() {}
