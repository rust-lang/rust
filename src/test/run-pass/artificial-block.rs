


// xfail-stage0
fn f() -> int { { ret 3; } }

fn main() { assert (f() == 3); }