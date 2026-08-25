#![feature(marker_trait_attr)]

#[marker] //~ ERROR attribute cannot be used on
struct Struct {}

#[marker] //~ ERROR attribute cannot be used on
impl Struct {}

#[marker] //~ ERROR attribute cannot be used on
union Union {
    x: i32,
}

#[marker] //~ ERROR attribute cannot be used on
const CONST: usize = 10;

#[marker] //~ ERROR attribute cannot be used on
fn function() {}

#[marker] //~ ERROR attribute cannot be used on
type Type = ();

fn main() {}
