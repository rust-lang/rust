// check-pass
#![feature(never_patterns)]
#![allow(incomplete_features)]
#![deny(unreachable_patterns)]

fn main() {}

enum Void {}

// A never pattern alone diverges.

fn never_arg(!: Void) -> u32 {}

fn ref_never_arg(&!: &Void) -> u32 {}

fn never_let() -> u32 {
    let ptr: *const Void = std::ptr::null();
    unsafe {
        let ! = *ptr;
    }
}

fn never_match() -> u32 {
    let ptr: *const Void = std::ptr::null();
    unsafe {
        match *ptr { ! };
    }
    println!(); // Ensures this typechecks because of divergence.
}
