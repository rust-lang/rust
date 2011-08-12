

// -*- rust -*-

fn some_vec(x: int) -> vec[int] { ret []; }

fn is_odd(n: int) -> bool { ret true; }

fn length_is_even(vs: vec[int]) -> bool { ret true; }

fn foo(acc: int, n: int) {
    if is_odd(n) && length_is_even(some_vec(1)) { log_err "bloop"; }
}

fn main() { foo(67, 5); }