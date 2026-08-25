// We used to bind the closure return type `&'a ()` with the late-bound vars of
// the owner (here `main` & `env` resp.) instead of the ones of the enclosing
// function-like / closure inside diagnostic code which was incorrect.

#![feature(closure_lifetime_binder)]

// issue: rust-lang/rust#130391
fn main() {
    let _ = for<'a> |x: &'a u8| -> &'a () { x }; //~ ERROR mismatched types
}

// issue: rust-lang/rust#130663
fn env<'r>() {
    let _ = for<'a> |x: &'a u8| -> &'a () { x }; //~ ERROR mismatched types
}
