//@ check-pass
//@ edition: 2018
#![feature(never_patterns)]
#![allow(incomplete_features)]
#![deny(unreachable_patterns)]

fn main() {}

enum Void {}

// A never pattern alone diverges.

fn never_arg(!: Void) -> ! {}

fn never_arg_returns_anything<T>(!: Void) -> T {}

fn ref_never_arg(&!: &Void) -> ! {}

fn never_let() -> ! {
    let ptr: *const Void = std::ptr::null();
    unsafe {
        let ! = *ptr;
    }
}

fn never_match() -> ! {
    let ptr: *const Void = std::ptr::null();
    unsafe {
        match *ptr { ! };
    }
    // Ensures this typechecks because of divergence and not the type of the match expression.
    println!();
}

// Note: divergence is not detected for async fns when the `!` is in the argument (#120240).
async fn async_let(x: Void) -> ! {
    let ! = x;
}
