#![feature(marker_trait_attr)]

#[marker] //~ ERROR attribute can only be applied to a trait
struct Struct {}

#[marker] //~ ERROR attribute can only be applied to a trait
impl Struct {}

#[marker] //~ ERROR attribute can only be applied to a trait
union Union {
    x: i32,
}

#[marker] //~ ERROR attribute can only be applied to a trait
const CONST: usize = 10;

#[marker] //~ ERROR attribute can only be applied to a trait
fn function() {}

#[marker] //~ ERROR attribute can only be applied to a trait
type Type = ();

fn main() {}
