//@ edition: 2018
//@ known-bug: #120240
#![feature(never_patterns)]
#![allow(incomplete_features)]

fn main() {}

enum Void {}

// Divergence is not detected.
async fn async_never(!: Void) -> ! {} // gives an error

// Divergence is detected
async fn async_let(x: Void) -> ! {
    let ! = x;
}
