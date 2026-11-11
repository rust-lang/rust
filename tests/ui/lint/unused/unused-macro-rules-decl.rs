#![feature(decl_macro)]
#![deny(unused_macro_rules)]
// To make sure we are not hitting this
#![deny(unused_macros)]

// Most simple case
macro num {
    (one) => { 1 },
    (two) => { 2 }, //~ ERROR: rule #2 of macro
    (three) => { 3 },
    (four) => { 4 }, //~ ERROR: rule #4 of macro
}
const _NUM: u8 = num!(one) + num!(three);

// Check that allowing the lint works
#[allow(unused_macro_rules)]
macro num_allowed {
    (one) => { 1 },
    (two) => { 2 },
    (three) => { 3 },
    (four) => { 4 },
}
const _NUM_ALLOWED: u8 = num_allowed!(one) + num_allowed!(three);

// Check that macro calls inside the macro trigger as usage
macro num_rec {
    (one) => { 1 },
    (two) => {
        num_rec!(one) + num_rec!(one)
    },
    (three) => { //~ ERROR: rule #3 of macro
        num_rec!(one) + num_rec!(two)
    },
    (four) => {
        num_rec!(two) + num_rec!(two)
    },
}
const _NUM_RECURSIVE: u8 = num_rec!(four);

// No error if the macro is public
pub macro num_public {
    (one) => { 1 },
    (two) => { 2 },
    (three) => { 3 },
    (four) => { 4 },
}
const _NUM_PUBLIC: u8 = num_public!(one) + num_public!(three);

fn main() {}
