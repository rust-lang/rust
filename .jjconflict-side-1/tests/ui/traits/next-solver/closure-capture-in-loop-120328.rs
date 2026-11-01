//@ compile-flags: -Znext-solver
//@ edition: 2021
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/120328>.
// Closures capturing an iterated variable in edition 2021 used to ICE
// with the next trait solver during MIR building.

fn main() {
    for item in &[1, 2, 3] {
        let _ = || *item;
    }
}
