#![allow(dead_code, clippy::double_parens, clippy::unnecessary_cast)]
#![warn(clippy::suboptimal_flops, clippy::imprecise_flops)]

// FIXME(f16_f128): add tests for these types once math functions are available

const TWO: f32 = 2.0;
const E: f32 = std::f32::consts::E;

fn check_log_base() {
    let x = 1f32;
    let _ = x.log(2f32);
    //~^ suboptimal_flops
    let _ = x.log(10f32);
    //~^ suboptimal_flops
    let _ = x.log(std::f32::consts::E);
    //~^ suboptimal_flops
    let _ = x.log(TWO);
    //~^ suboptimal_flops
    let _ = x.log(E);
    //~^ suboptimal_flops
    let _ = (x as f32).log(2f32);
    //~^ suboptimal_flops

    let x = 1f64;
    let _ = x.log(2f64);
    //~^ suboptimal_flops
    let _ = x.log(10f64);
    //~^ suboptimal_flops
    let _ = x.log(std::f64::consts::E);
    //~^ suboptimal_flops
}

fn check_ln1p() {
    let x = 1f32;
    let _ = (1f32 + 2.).ln();
    //~^ imprecise_flops
    let _ = (1f32 + 2.0).ln();
    //~^ imprecise_flops
    let _ = (1.0 + x).ln();
    //~^ imprecise_flops
    let _ = (1.0 + x / 2.0).ln();
    //~^ imprecise_flops
    let _ = (1.0 + x.powi(3)).ln();
    //~^ imprecise_flops
    let _ = (1.0 + x.powi(3) / 2.0).ln();
    //~^ imprecise_flops
    let _ = (1.0 + (std::f32::consts::E - 1.0)).ln();
    //~^ imprecise_flops
    let _ = (x + 1.0).ln();
    //~^ imprecise_flops
    let _ = (x.powi(3) + 1.0).ln();
    //~^ imprecise_flops
    let _ = (x + 2.0 + 1.0).ln();
    //~^ imprecise_flops
    let _ = (x / 2.0 + 1.0).ln();
    //~^ imprecise_flops
    // Cases where the lint shouldn't be applied
    let _ = (1.0 + x + 2.0).ln();
    let _ = (x + 1.0 + 2.0).ln();
    let _ = (x + 1.0 / 2.0).ln();
    let _ = (1.0 + x - 2.0).ln();

    let x = 1f64;
    let _ = (1f64 + 2.).ln();
    //~^ imprecise_flops
    let _ = (1f64 + 2.0).ln();
    //~^ imprecise_flops
    let _ = (1.0 + x).ln();
    //~^ imprecise_flops
    let _ = (1.0 + x / 2.0).ln();
    //~^ imprecise_flops
    let _ = (1.0 + x.powi(3)).ln();
    //~^ imprecise_flops
    let _ = (x + 1.0).ln();
    //~^ imprecise_flops
    let _ = (x.powi(3) + 1.0).ln();
    //~^ imprecise_flops
    let _ = (x + 2.0 + 1.0).ln();
    //~^ imprecise_flops
    let _ = (x / 2.0 + 1.0).ln();
    //~^ imprecise_flops
    // Cases where the lint shouldn't be applied
    let _ = (1.0 + x + 2.0).ln();
    let _ = (x + 1.0 + 2.0).ln();
    let _ = (x + 1.0 / 2.0).ln();
    let _ = (1.0 + x - 2.0).ln();
}

fn issue12881() {
    pub trait MyLog {
        fn log(&self) -> Self;
    }

    impl MyLog for f32 {
        fn log(&self) -> Self {
            4.
        }
    }

    let x = 2.0;
    x.log();
}

fn main() {}
