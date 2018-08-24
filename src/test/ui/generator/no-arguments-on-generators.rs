#![feature(generators)]

fn main() {
    let gen = |start| { //~ ERROR generators cannot have explicit arguments
        yield;
    };
}
