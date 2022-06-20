fn unsat_safe_1(x: i32) -> () {
    assert!(x <= -200 || x >= -300);
}

fn unsat_safe_2(x: i32) -> () {
    assert!(x == x);
}

fn sat_unsafe_1(x: i32) -> () {
    assert!(x < 13);
}

fn sat_unsafe_2(x: i32) -> () {
    assert!(x <= -200 && x >= -300);
}

fn main() {
    unsat_safe_1(12);
    unsat_safe_2(14);
    sat_unsafe_1(12);
    sat_unsafe_2(-250);
}
