//@ edition: 2024
#![feature(never_patterns)]
#![allow(incomplete_features)]
#![deny(unreachable_patterns)]

fn main() {}

enum Void {}

// Contrast with `./diverges.rs`: merely having an empty type around isn't enough to diverge.

fn wild_void(_: Void) -> u32 {}
//~^ ERROR: mismatched types

fn wild_let() -> u32 {
    let ptr: *const Void = std::ptr::null();
    unsafe {
        //~^ ERROR: mismatched types
        let _ = *ptr;
    }
}

fn wild_match() -> u32 {
    let ptr: *const Void = std::ptr::null();
    unsafe {
        match *ptr {
            _ => {} //~ ERROR: mismatched types
        }
    }
}

fn binding_void(_x: Void) -> u32 {}
//~^ ERROR: mismatched types

fn binding_let() -> u32 {
    let ptr: *const Void = std::ptr::null();
    unsafe {
        //~^ ERROR: mismatched types
        let _x = *ptr;
    }
}

fn binding_match() -> u32 {
    let ptr: *const Void = std::ptr::null();
    unsafe {
        match *ptr {
            _x => {} //~ ERROR: mismatched types
        }
    }
}

// Don't confuse this with a `let !` statement.
fn let_chain(x: Void) -> u32 {
    if let true = true && let ! = x {}
    //~^ ERROR: mismatched types
}
