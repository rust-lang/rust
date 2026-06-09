#![warn(clippy::suboptimal_flops, clippy::imprecise_flops)]
#![allow(clippy::unnecessary_cast)]

// FIXME(f16_f128): add tests for these types when `powf` is available

fn main() {
    let x = 3f32;
    let _ = 2f32.powf(x);
    //~^ suboptimal_flops
    let _ = 2f32.powf(3.1);
    //~^ suboptimal_flops
    let _ = 2f32.powf(-3.1);
    //~^ suboptimal_flops
    let _ = std::f32::consts::E.powf(x);
    //~^ suboptimal_flops
    let _ = std::f32::consts::E.powf(3.1);
    //~^ suboptimal_flops
    let _ = std::f32::consts::E.powf(-3.1);
    //~^ suboptimal_flops
    let _ = x.powf(1.0 / 2.0);
    //~^ suboptimal_flops
    let _ = x.powf(1.0 / 3.0);
    //~^ imprecise_flops
    let _ = (x as f32).powf(1.0 / 3.0);
    //~^ imprecise_flops
    let _ = x.powf(3.0);
    //~^ suboptimal_flops
    let _ = x.powf(-2.0);
    //~^ suboptimal_flops
    let _ = x.powf(16_777_215.0);
    //~^ suboptimal_flops
    let _ = x.powf(-16_777_215.0);
    //~^ suboptimal_flops
    let _ = (x as f32).powf(-16_777_215.0);
    //~^ suboptimal_flops
    let _ = (x as f32).powf(3.0);
    //~^ suboptimal_flops
    let _ = (1.5_f32 + 1.0).powf(1.0 / 3.0);
    //~^ imprecise_flops
    let _ = 1.5_f64.powf(1.0 / 3.0);
    //~^ imprecise_flops
    let _ = 1.5_f64.powf(1.0 / 2.0);
    //~^ suboptimal_flops
    let _ = 1.5_f64.powf(3.0);
    //~^ suboptimal_flops

    macro_rules! m {
        ($e:expr) => {
            5.5 - $e
        };
    }

    let _ = 2f32.powf(1f32 + m!(2.0));
    //~^ suboptimal_flops

    // Cases where the lint shouldn't be applied
    let _ = x.powf(2.1);
    let _ = x.powf(-2.1);
    let _ = x.powf(16_777_216.0);
    let _ = x.powf(-16_777_216.0);

    let x = 3f64;
    let _ = 2f64.powf(x);
    //~^ suboptimal_flops
    let _ = 2f64.powf(3.1);
    //~^ suboptimal_flops
    let _ = 2f64.powf(-3.1);
    //~^ suboptimal_flops
    let _ = std::f64::consts::E.powf(x);
    //~^ suboptimal_flops
    let _ = std::f64::consts::E.powf(3.1);
    //~^ suboptimal_flops
    let _ = std::f64::consts::E.powf(-3.1);
    //~^ suboptimal_flops
    let _ = x.powf(1.0 / 2.0);
    //~^ suboptimal_flops
    let _ = x.powf(1.0 / 3.0);
    //~^ imprecise_flops
    let _ = x.powf(3.0);
    //~^ suboptimal_flops
    let _ = x.powf(-2.0);
    //~^ suboptimal_flops
    let _ = x.powf(-2_147_483_648.0);
    //~^ suboptimal_flops
    let _ = x.powf(2_147_483_647.0);
    //~^ suboptimal_flops
    // Cases where the lint shouldn't be applied
    let _ = x.powf(2.1);
    let _ = x.powf(-2.1);
    let _ = x.powf(-2_147_483_649.0);
    let _ = x.powf(2_147_483_648.0);
}
