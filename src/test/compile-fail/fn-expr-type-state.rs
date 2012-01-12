// error-pattern:Unsatisfied precondition

fn main() {
    // Typestate should work even in a fn@. we should reject this program.
    let f = fn@() -> int { let i: int; ret i; };
    log(error, f());
}
