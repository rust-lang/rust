// error-pattern:Variable 'x' captured more than once
fn main() {
    let x = 5;
    let y = sendfn[move x; copy x]() -> int { x };
}
