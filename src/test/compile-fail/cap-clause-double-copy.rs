// error-pattern:Variable 'x' captured more than once
fn main() {
    let x = 5;
    let y = fn~[copy x, x]() -> int { x };
}
