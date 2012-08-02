// -*- rust -*-
// error-pattern: not all control paths return a value

fn god_exists(a: int) -> bool { return god_exists(a); }

fn f(a: int) -> int { if god_exists(a) { return 5; }; }

fn main() { f(12); }
