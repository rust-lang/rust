#![feature(generators)]

fn main() {
    let gen = |start| {
        //~^ ERROR type inside generator must be known in this context
        yield;
        //~^ ERROR type inside generator must be known in this context
        //~| ERROR type inside generator must be known in this context
    };
}
