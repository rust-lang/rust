


// -*- rust -*-
fn f<T: Copy, U: Copy>(x: T, y: U) -> {a: T, b: U} { return {a: x, b: y}; }

fn main() {
    log(debug, f({x: 3, y: 4, z: 5}, 4).a.x);
    log(debug, f(5, 6).a);
}
