// error-pattern:Upvars (like 'x') cannot be moved into a closure
fn main() {
    let x = 5;
    let _y = fn~[move x]() -> int {
        let _z = fn~[move x]() -> int { x };
        22
    };
}
