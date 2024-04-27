#[warn(clippy::double_neg)]
#[allow(clippy::no_effect)]
fn main() {
    let x = 1;
    -x;
    -(-x);
    --x;
    //~^ ERROR: `--x` could be misinterpreted as pre-decrement by C programmers, is usuall
    //~| NOTE: `-D clippy::double-neg` implied by `-D warnings`
}
