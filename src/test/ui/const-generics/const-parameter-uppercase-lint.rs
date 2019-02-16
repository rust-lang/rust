#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

#![deny(non_upper_case_globals)]

fn noop<const x: u32>() {
    //~^ ERROR const generics in any position are currently unsupported
}
