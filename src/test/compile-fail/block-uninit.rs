// error-pattern: Unsatisfied precondition constraint
// xfail-stage0

fn force(f: &block() ) { f(); }
fn main() { let x: int; force(block () { log_err x; }); }