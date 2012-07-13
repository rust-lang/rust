// error-pattern:variable `x` captured more than once
fn main() {
    let x = 5;
    let y = fn~(move x, move x) -> int { x };
}
