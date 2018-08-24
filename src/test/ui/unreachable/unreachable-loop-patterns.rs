#![feature(never_type)]
#![feature(exhaustive_patterns)]
#![deny(unreachable_patterns)]

fn main() {
    let x: &[!] = &[];

    for _ in x {}
    //~^ ERROR unreachable pattern
}

