#![feature(plugin)]
#![plugin(clippy)]

#[deny(double_neg)]
fn main() {
    let x = 1;
    -x;
    -(-x);
    --x; //~ERROR: `--x` could be misinterpreted as pre-decrement by C programmers, is usually a no-op
}
