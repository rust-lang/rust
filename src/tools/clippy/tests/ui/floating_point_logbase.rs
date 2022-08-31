// run-rustfix
#![warn(clippy::suboptimal_flops)]

fn main() {
    let x = 3f32;
    let y = 5f32;
    let _ = x.ln() / y.ln();
    let _ = (x as f32).ln() / y.ln();
    let _ = x.log2() / y.log2();
    let _ = x.log10() / y.log10();
    let _ = x.log(5f32) / y.log(5f32);
    // Cases where the lint shouldn't be applied
    let _ = x.ln() / y.powf(3.2);
    let _ = x.powf(3.2) / y.powf(3.2);
    let _ = x.powf(3.2) / y.ln();
    let _ = x.log(5f32) / y.log(7f32);
}
