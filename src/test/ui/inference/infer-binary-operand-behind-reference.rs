// check-pass

fn main() {
    let _: u8 = 0 + 0;
    let _: u8 = 0 + &0;
    let _: u8 = &0 + 0;
    let _: u8 = &0 + &0;

    let _: f32 = 0.0 + 0.0;
    let _: f32 = 0.0 + &0.0;
    let _: f32 = &0.0 + 0.0;
    let _: f32 = &0.0 + &0.0;
}
