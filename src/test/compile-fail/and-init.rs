// error-pattern: Unsatisfied precondition constraint (for example, init(i

fn main() {
    let i: int;

    log(debug, false && { i = 5; true });
    log(debug, i);
}
