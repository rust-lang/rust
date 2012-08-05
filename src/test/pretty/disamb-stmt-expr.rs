// pp-exact

// Here we check that the parentheses around the body of `wsucc()` are
// preserved.  They are needed to disambiguate `{return n+1}; - 0` from
// `({return n+1}-0)`.

fn id(f: fn&() -> int) -> int { f() }

fn wsucc(n: int) -> int { (do id || { 1 }) - 0 }
fn main() { }
