#![deny(unused_macro_rules)]
// To make sure we are not hitting this
#![deny(unused_macros)]

// Most simple case
macro_rules! num {
    (one) => { 1 };
    (two) => { 2 }; //~ ERROR: 2nd rule of macro
    (three) => { 3 };
    (four) => { 4 }; //~ ERROR: 4th rule of macro
}
const _NUM: u8 = num!(one) + num!(three);

// Check that allowing the lint works
#[allow(unused_macro_rules)]
macro_rules! num_allowed {
    (one) => { 1 };
    (two) => { 2 };
    (three) => { 3 };
    (four) => { 4 };
}
const _NUM_ALLOWED: u8 = num_allowed!(one) + num_allowed!(three);

// Check that macro calls inside the macro trigger as usage
macro_rules! num_rec {
    (one) => { 1 };
    (two) => {
        num_rec!(one) + num_rec!(one)
    };
    (three) => { //~ ERROR: 3rd rule of macro
        num_rec!(one) + num_rec!(two)
    };
    (four) => { num_rec!(two) + num_rec!(two) };
}
const _NUM_RECURSIVE: u8 = num_rec!(four);

// No error if the macro is being exported
#[macro_export]
macro_rules! num_exported {
    (one) => { 1 };
    (two) => { 2 };
    (three) => { 3 };
    (four) => { 4 };
}
const _NUM_EXPORTED: u8 = num_exported!(one) + num_exported!(three);

fn main() {}
