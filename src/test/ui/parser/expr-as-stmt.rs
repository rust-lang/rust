// run-rustfix
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_must_use)]

fn foo() -> i32 {
    {2} + {2} //~ ERROR expected expression, found `+`
    //~^ ERROR mismatched types
}

fn bar() -> i32 {
    {2} + 2 //~ ERROR expected expression, found `+`
    //~^ ERROR mismatched types
}

fn zul() -> u32 {
    let foo = 3;
    { 42 } + foo; //~ ERROR expected expression, found `+`
    //~^ ERROR mismatched types
    32
}

fn baz() -> i32 {
    { 3 } * 3 //~ ERROR type `{integer}` cannot be dereferenced
    //~^ ERROR mismatched types
}

fn qux(a: Option<u32>, b: Option<u32>) -> bool {
    if let Some(x) = a { true } else { false }
    && //~ ERROR expected expression
    if let Some(y) = a { true } else { false }
}

fn moo(x: u32) -> bool {
    match x {
        _ => 1,
    } > 0 //~ ERROR expected expression
}

fn main() {}
