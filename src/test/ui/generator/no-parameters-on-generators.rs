#![feature(generators)]

fn main() {
    let gen = |start| { //~ ERROR generators cannot have explicit parameters
        yield;
    };
}
