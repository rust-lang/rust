#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

#![deny(non_upper_case_globals)]

fn noop<const x: u32>() {
    //~^ ERROR const parameter `x` should have an upper case name
}

fn main() {}
