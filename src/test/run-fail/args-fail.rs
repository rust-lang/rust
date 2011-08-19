// error-pattern:meep
fn f(a: int, b: int, c: @int) { fail "moop"; }

fn main() { f(1, fail "meep", @42); }
