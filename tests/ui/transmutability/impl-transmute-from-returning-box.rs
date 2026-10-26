// Regression test for: https://github.com/rust-lang/rust/issues/154536
//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [next] compile-flags: -Znext-solver

#![feature(transmutability)]
#![feature(unboxed_closures)]

pub fn box_factory() -> impl std::mem::TransmuteFrom() {
    //~^ ERROR: `()` cannot be safely transmuted into `Box<()>`
    //~^^ ERROR: associated type `Output` not found for `TransmuteFrom`
    //~^^^ ERROR: associated type `Output` not found for `TransmuteFrom`
    Box::new(())
}

fn main() {}
