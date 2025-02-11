#![warn(clippy::suboptimal_flops)]
#![allow(clippy::unnecessary_cast)]

fn main() {
    let x = 3f32;
    let y = 5f32;
    let _ = x.ln() / y.ln();
    //~^ suboptimal_flops
    let _ = (x as f32).ln() / y.ln();
    //~^ suboptimal_flops
    let _ = x.log2() / y.log2();
    //~^ suboptimal_flops
    let _ = x.log10() / y.log10();
    //~^ suboptimal_flops
    let _ = x.log(5f32) / y.log(5f32);
    //~^ suboptimal_flops
    // Cases where the lint shouldn't be applied
    let _ = x.ln() / y.powf(3.2);
    let _ = x.powf(3.2) / y.powf(3.2);
    let _ = x.powf(3.2) / y.ln();
    let _ = x.log(5f32) / y.log(7f32);
}
