// error-pattern: Unsatisfied precondition constraint

fn force(f: block()) { f(); }
fn main() { let x: int; force(block () { log_err x; }); }
