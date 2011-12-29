// pp-exact

// Here we check that the parentheses around the body of `wsucc()` are
// preserved.  They are needed to disambiguate `{ret n+1}; - 0` from
// `({ret n+1}-0)`.

fn wsucc(n: int) -> int { ({ ret n + 1 } - 0); }
fn main() { }
