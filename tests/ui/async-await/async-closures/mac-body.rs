//@ edition: 2021
//@ check-pass

#![feature(async_closure)]

// Make sure we don't ICE if an async closure has a macro body.
// This happened because we were calling walk instead of visit
// in the def collector, oops!

fn main() {
    let _ = async || println!();
}
