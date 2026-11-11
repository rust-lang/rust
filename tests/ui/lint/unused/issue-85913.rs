#![deny(unused_must_use)]

pub fn fun() -> i32 {
    function() && return 1;
    //~^ ERROR: unused logical operation that must be used
    return 0;
}

fn function() -> bool {
    true
}

fn main() {}
