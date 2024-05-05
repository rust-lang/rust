#[deny(clippy::while_float)]
fn main() {
    let mut x = 0.0_f32;
    while x < 42.0_f32 {
        x += 0.5;
    }
    while x < 42.0 {
        x += 1.0;
    }
    let mut x = 0;
    while x < 42 {
        x += 1;
    }
}
