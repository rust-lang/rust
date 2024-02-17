//@ compile-flags: --extern čɍαţē=libnon_ascii.rlib
//@ error-pattern: crate name `čɍαţē` passed to `--extern` is not a valid ASCII identifier

fn main() {}
