#![allow(dead_code)]
#![deny(unused_parens)]

enum State {
    Waiting { start_at: u64 }
}
struct Foo {}

fn main() {
    let e = &mut State::Waiting { start_at: 0u64 };
    match (&mut State::Waiting { start_at: 0u64 }) {
        _ => {}
    }

    match (e) {
        //~^ ERROR unnecessary parentheses around `match` scrutinee expression
        _ => {}
    }

    match &(State::Waiting { start_at: 0u64 }) {
        _ => {}
    }

    match (State::Waiting { start_at: 0u64 }) {
        _ => {}
    }

    match (&&Foo {}) {
        _ => {}
    }
}
