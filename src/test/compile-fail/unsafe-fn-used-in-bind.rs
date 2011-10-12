// -*- rust -*-
// error-pattern: unsafe functions can only be called

unsafe fn f(x: int, y: int) -> int { ret x + y; }

fn main() {
    let x = bind f(3, _);
    let y = x(4);
}
