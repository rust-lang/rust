//@ known-bug: rust-lang/rust#130399

fn elided(main: &()) -> impl Sized + use<main> {}

fn main() {}
