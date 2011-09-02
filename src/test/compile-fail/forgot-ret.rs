// -*- rust -*-
// error-pattern: mismatched types

fn god_exists(a: int) -> bool { be god_exists(a); }

fn f(a: int) -> int { if god_exists(a) { ret 5; } }

fn main() { f(12); }
