#![allow(dead_code)]

fn main() {}

struct Struct;
trait Trait {}
impl Trait for Struct {}
impl Trait for u32 {}
impl Trait for () {}

fn implicit_return() -> dyn Trait {
    //~^ ERROR E0746
    Struct
}

fn return_in_branch() -> dyn Trait {
    //~^ ERROR E0746
    if true {
        return 0;
    }
    42
}

// The examples below are mostly here to test the diagnostic output,
// in particular whether the suggestion is valid.

fn return_in_tail_loop(c: bool) -> dyn Trait {
    //~^ ERROR E0746
    loop {
        if c {
            return 1u32;
        }
    }
}

fn return_nested_in_leaf(c: bool) -> dyn Trait {
    //~^ ERROR E0746
    let identity = |x| x;
    identity(if c { return 1u32 } else { 2u32 })
}

fn labeled_block_break() -> dyn Trait {
    //~^ ERROR E0746
    'a: {
        break 'a 1u32; //~ ERROR mismatched types
    }
}

fn loop_break_value() -> dyn Trait {
    //~^ ERROR E0746
    loop {
        break 1u32; //~ ERROR mismatched types
    }
}
