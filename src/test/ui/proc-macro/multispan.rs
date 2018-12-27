// aux-build:multispan.rs

#![feature(proc_macro_hygiene)]

extern crate multispan;

use multispan::hello;

fn main() {
    // This one emits no error.
    hello!();

    // Exactly one 'hi'.
    hello!(hi); //~ ERROR hello to you, too!

    // Now two, back to back.
    hello!(hi hi); //~ ERROR hello to you, too!

    // Now three, back to back.
    hello!(hi hi hi); //~ ERROR hello to you, too!

    // Now several, with spacing.
    hello!(hi hey hi yo hi beep beep hi hi); //~ ERROR hello to you, too!
    hello!(hi there, hi how are you? hi... hi.); //~ ERROR hello to you, too!
    hello!(whoah. hi di hi di ho); //~ ERROR hello to you, too!
    hello!(hi good hi and good bye); //~ ERROR hello to you, too!
}
