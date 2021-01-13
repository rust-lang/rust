// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

#![deny(non_upper_case_globals)]

fn noop<const x: u32>() {
    //~^ ERROR const parameter `x` should have an upper case name
}

fn main() {}
