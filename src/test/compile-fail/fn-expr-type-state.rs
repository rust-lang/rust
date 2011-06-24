// error-pattern:Unsatisfied precondition
// xfail-stage0

fn main() {
    // Typestate should work even in a lambda. we should reject this program.
    auto f = fn () -> int {
        let int i;
        ret i;
    };
    log_err f();
}
