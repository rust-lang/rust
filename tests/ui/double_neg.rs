#[warn(clippy::double_neg)]
fn main() {
    let x = 1;
    -x;
    -(-x);
    --x;
}
