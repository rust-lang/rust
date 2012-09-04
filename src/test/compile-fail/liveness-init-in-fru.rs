// -*- rust -*-

type point = {x: int, y: int};

fn main() {
    let mut origin: point;
    origin = {x: 10,.. origin}; //~ ERROR use of possibly uninitialized variable: `origin`
    copy origin;
}
