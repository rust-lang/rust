#[allow(clippy::no_effect)]
#[warn(clippy::erasing_op)]
fn main() {
    let x: u8 = 0;

    x * 0;
    0 & x;
    0 / x;
}
