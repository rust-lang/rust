#![feature(never_patterns)]
#![allow(incomplete_features)]
#![deny(unreachable_patterns)]
#![deny(unreachable_code)]

fn main() {}

enum Void {}

fn never_arg(!: Void) -> u32 {
    println!();
    //~^ ERROR unreachable statement
}

fn ref_never_arg(&!: &Void) -> u32 {
    println!();
    //~^ ERROR unreachable statement
}

fn never_let() -> u32 {
    let ptr: *const Void = std::ptr::null();
    unsafe {
        let ! = *ptr;
    }
    println!();
    //~^ ERROR unreachable statement
}

fn never_match() -> u32 {
    let ptr: *const Void = std::ptr::null();
    unsafe {
        match *ptr { ! };
    }
    println!();
    //~^ ERROR unreachable statement
}
