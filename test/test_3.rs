fn unsat_safe_1(x: i32) -> () {
    assert!(x <= -200 || x >= -300);
}

fn sat_unsafe_1(x: i32) -> () {
    assert!(x < 13);
}

fn main() {
    unsat_safe_1(12);
    sat_unsafe_1(12);
}
