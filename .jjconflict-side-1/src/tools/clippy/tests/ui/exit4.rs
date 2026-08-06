//@ check-pass
//@compile-flags: --test

#![warn(clippy::exit)]

fn main() {
    std::process::exit(0)
}
