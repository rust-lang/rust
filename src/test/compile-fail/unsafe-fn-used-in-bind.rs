// -*- rust -*-

unsafe fn f(x: int, y: int) -> int { ret x + y; }

fn main() {
    let x = bind f(3, _);
    //!^ ERROR access to unsafe function requires unsafe function or block
    let y = x(4);
}
