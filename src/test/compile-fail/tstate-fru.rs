// -*- rust -*-

type point = {x: int, y: int};

pure fn test(_p: point) -> bool { true }
fn tested(p: point) : test(p) -> point { p }

fn main() {
    let origin: point;
    origin = {x: 0, y: 0};
    let right: point = {x: 10 with tested(origin)};
        //!^ ERROR precondition
    copy right;
}
