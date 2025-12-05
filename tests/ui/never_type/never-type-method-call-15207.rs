//! Regression test for https://github.com/rust-lang/rust/issues/15207

fn main() {
    loop {
        break.push(1) //~ ERROR no method named `push` found for type `!`
        ;
    }
}
