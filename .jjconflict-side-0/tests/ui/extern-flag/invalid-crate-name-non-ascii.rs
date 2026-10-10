//@ compile-flags: --extern čɍαţē=libnon_ascii.rlib

fn main() {}

//~? ERROR crate name `čɍαţē` passed to `--extern` is not a valid ASCII identifier
