// -*- rust -*-

// error-pattern: expected the constraint name

obj f() {
    fn g(q: int) -> bool { ret true; }
}

fn main() {
    let z = f();
    // should fail to typecheck, as z.g isn't an explicit name
    check (z.g(42));
}
