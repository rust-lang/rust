


#[allow(no_effect)]
#[warn(erasing_op)]
fn main() {
    let x: u8 = 0;

    x * 0;
    0 & x;
    0 / x;
}
