


// -*- rust -*-
fn f<T: copy, U: copy>(x: T, y: U) -> {a: T, b: U} { ret {a: x, b: y}; }

fn main() {
    log(debug, f({x: 3, y: 4, z: 5}, 4).a.x);
    log(debug, f(5, 6).a);
}
