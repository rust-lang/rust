//@ proc-macro: multispan.rs
//@ compile-flags: --error-format human-annotate-rs -Z unstable-options

#![feature(proc_macro_hygiene)]

extern crate multispan;

use multispan::hello;

fn main() {
    // This one emits no error.
    hello!();

    // Exactly one 'hi'.
    hello!(hi);

    // Now two, back to back.
    hello!(hi hi);

    // Now three, back to back.
    hello!(hi hi hi);

    // Now several, with spacing.
    hello!(hi hey hi yo hi beep beep hi hi);
    hello!(hi there, hi how are you? hi... hi.);
    hello!(whoah. hi di hi di ho);
    hello!(hi good hi and good bye);
}

//~? RAW hello to you, too!
