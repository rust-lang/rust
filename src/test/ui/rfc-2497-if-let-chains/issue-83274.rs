// check-fail

pub fn main() {
    let x = Some(3);
    if (let Some(y) = x) { //~ ERROR invalid parentheses around `let` expression in `if let`
    }
}
