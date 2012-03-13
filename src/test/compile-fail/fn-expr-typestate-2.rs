// error-pattern:unsatisfied precondition

fn main() {
    let j = fn@() -> int { let i: int; ret i; }();
    log(error, j);
}
