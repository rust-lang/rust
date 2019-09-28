#![feature(generators)]

fn main() {
    let gen = |start| { //~ ERROR generators cannot have explicit parameters
        //~^ ERROR type inside generator must be known in this context
        yield;
    };
}
