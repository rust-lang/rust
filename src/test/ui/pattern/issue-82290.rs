// check-pass

#![feature(let_chains)] //~ WARN the feature `let_chains` is incomplete

fn main() {
    if true && let x = 1 { //~ WARN irrefutable `let` pattern
        let _ = x;
    }
}
