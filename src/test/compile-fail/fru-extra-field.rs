// -*- rust -*-

// error-pattern: record

type point = {x: int, y: int};

fn main() {
    let origin: point = {x: 0, y: 0};

    let origin3d: point = {z: 0 with origin};
}
