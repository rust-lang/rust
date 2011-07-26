// error-pattern: Unsatisfied precondition constraint
// xfail-stage0

fn force(&block() f) { f(); }
fn main() {
    let int x;
    force(block() { log_err x; });
}
