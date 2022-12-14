#[warn(clippy::double_neg)]
#[allow(clippy::no_effect)]
fn main() {
    let x = 1;
    -x;
    -(-x);
    --x;
}
