//@ proc-macro: respan.rs
//@ revisions: rpass1 rpass2

extern crate respan;

#[macro_use]
#[path = "invalid-span-helper-mod.rs"]
mod invalid_span_helper_mod;

// Invoke a macro from a different file - this
// allows us to get tokens with spans from different files
helper!(1);
