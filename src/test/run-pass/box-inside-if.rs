

// -*- rust -*-

fn some_box(x: int) -> @int { ret @x; }

fn is_odd(n: int) -> bool { ret true; }

fn length_is_even(vs: @int) -> bool { ret true; }

fn foo(acc: int, n: int) {
    if is_odd(n) && length_is_even(some_box(1)) { #error("bloop"); }
}

fn main() { foo(67, 5); }
