// error-pattern:error: Variable 'x' captured more than once
fn main() {
    let x = 5;
    let y = sendfn[copy x, x]() -> int { x };
}
