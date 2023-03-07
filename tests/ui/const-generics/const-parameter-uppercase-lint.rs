#![deny(non_upper_case_globals)]

fn noop<const x: u32>() {
    //~^ ERROR const parameter `x` should have an upper case name
}

fn main() {}
