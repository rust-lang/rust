// -*- rust -*-

// error-pattern: precondition

type point = {x: int, y: int};

fn main() {
    let origin: point;

    let right: point = {x: 10 with origin};
    origin = {x: 0, y: 0};
}