// -*- rust -*-
// error-pattern: not all control paths return

fn god_exists(a: int) -> bool { be god_exists(a); }

fn f(a: int) -> int { if god_exists(a) { ret 5; } }

fn main() { f(12); }