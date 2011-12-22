


// -*- rust -*-
fn f<copy T, copy U>(x: T, y: U) -> {a: T, b: U} { ret {a: x, b: y}; }

fn main() {
    log_full(core::debug, f({x: 3, y: 4, z: 5}, 4).a.x);
    log_full(core::debug, f(5, 6).a);
}
