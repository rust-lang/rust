//@ check-pass
//@compile-flags: --test

#![warn(clippy::exit)]

fn main() {
    if true {
        std::process::exit(2);
    };
    std::process::exit(1);
}
