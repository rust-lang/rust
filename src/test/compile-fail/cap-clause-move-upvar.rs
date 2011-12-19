// error-pattern: error: Upvars (like 'x') cannot be moved into a closure
fn main() {
    let x = 5;
    let _y = sendfn[move x]() -> int {
        let _z = sendfn[move x]() -> int { x };
        22
    };
}
