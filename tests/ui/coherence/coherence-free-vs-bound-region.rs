// Capture a coherence pattern from wasm-bindgen that we discovered as part of
// future-compatibility warning #56105. This pattern currently receives a lint
// warning but we probably want to support it long term.
//
// Key distinction: we are implementing once for `A` (take ownership) and one
// for `&A` (borrow).
//
// c.f. #56105

#![deny(coherence_leak_check)]

trait TheTrait {}

impl<'a> TheTrait for fn(&'a u8) {}

impl TheTrait for fn(&u8) {
    //~^ ERROR conflicting implementations of trait
    //~| WARN the behavior may change in a future release
}

fn main() {}
