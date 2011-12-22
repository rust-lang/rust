// error-pattern: Unsatisfied precondition constraint (for example, init(i

fn main() {
    let i: int;

    log_full(core::debug, false && { i = 5; true });
    log_full(core::debug, i);
}
