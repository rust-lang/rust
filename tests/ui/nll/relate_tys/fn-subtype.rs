// Test that NLL produces correct spans for higher-ranked subtyping errors.
//
//@ compile-flags:-Zno-leak-check

fn main() {
    let x: fn(&'static ()) = |_| {};
    let y: for<'a> fn(&'a ()) = x; //~ ERROR mismatched types [E0308]
}
