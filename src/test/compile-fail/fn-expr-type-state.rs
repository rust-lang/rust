// error-pattern:Unsatisfied precondition
// xfail-stage0

fn main() {
    // Typestate should work even in a lambda. we should reject this program.
    let f = fn () -> int { let i: int; ret i; };
    log_err f();
}