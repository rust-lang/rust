#![feature(let_chains)] //~ WARN the feature `let_chains` is incomplete

fn main() {
    if true && let x = 1 { //~ ERROR `let` expressions are not supported here
        let _ = x;
    }
}
