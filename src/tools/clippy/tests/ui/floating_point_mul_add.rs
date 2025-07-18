#![warn(clippy::suboptimal_flops)]

/// Allow suboptimal_ops in constant context
pub const fn in_const_context() {
    let a: f64 = 1234.567;
    let b: f64 = 45.67834;
    let c: f64 = 0.0004;

    let _ = a * b + c;
    let _ = c + a * b;
}

fn main() {
    let a: f64 = 1234.567;
    let b: f64 = 45.67834;
    let c: f64 = 0.0004;
    let d: f64 = 0.0001;

    let _ = a * b + c;
    //~^ suboptimal_flops
    let _ = a * b - c;
    //~^ suboptimal_flops
    let _ = c + a * b;
    //~^ suboptimal_flops
    let _ = c - a * b;
    //~^ suboptimal_flops
    let _ = a + 2.0 * 4.0;
    //~^ suboptimal_flops
    let _ = a + 2. * 4.;
    //~^ suboptimal_flops

    let _ = (a * b) + c;
    //~^ suboptimal_flops
    let _ = c + (a * b);
    //~^ suboptimal_flops
    let _ = a * b * c + d;
    //~^ suboptimal_flops

    let _ = a.mul_add(b, c) * a.mul_add(b, c) + a.mul_add(b, c) + c;
    //~^ suboptimal_flops
    let _ = 1234.567_f64 * 45.67834_f64 + 0.0004_f64;
    //~^ suboptimal_flops

    let _ = (a * a + b).sqrt();
    //~^ suboptimal_flops

    let u = 1usize;
    let _ = a - (b * u as f64);
    //~^ suboptimal_flops

    // Cases where the lint shouldn't be applied
    let _ = (a * a + b * b).sqrt();
}

fn _issue11831() {
    struct NotAFloat;

    impl std::ops::Add<f64> for NotAFloat {
        type Output = Self;

        fn add(self, _: f64) -> Self {
            NotAFloat
        }
    }

    let a = NotAFloat;
    let b = 1.0_f64;
    let c = 1.0;

    let _ = a + b * c;
}

fn _issue14897() {
    let x = 1.0;
    let _ = x * 2.0 + 0.5; // should not suggest mul_add
    let _ = 0.5 + x * 2.0; // should not suggest mul_add
    let _ = 0.5 + x * 1.2; // should not suggest mul_add
    let _ = 1.2 + x * 1.2; // should not suggest mul_add

    let x = -1.0;
    let _ = 0.5 + x * 1.2; // should not suggest mul_add

    let x = { 4.0 };
    let _ = 0.5 + x * 1.2; // should not suggest mul_add

    let x = if 1 > 2 { 1.0 } else { 2.0 };
    let _ = 0.5 + x * 1.2; // should not suggest mul_add

    let x = 2.4 + 1.2;
    let _ = 0.5 + x * 1.2; // should not suggest mul_add

    let f = || 4.0;
    let x = f();
    let _ = 0.5 + f() * 1.2; // should not suggest mul_add
    let _ = 0.5 + x * 1.2; // should not suggest mul_add

    let x = 0.1;
    let y = x;
    let z = y;
    let _ = 0.5 + z * 1.2; // should not suggest mul_add

    let _ = 0.5 + 2.0 * x;
    //~^ suboptimal_flops
    let _ = 2.0 * x + 0.5;
    //~^ suboptimal_flops

    let _ = x + 2.0 * 4.0;
    //~^ suboptimal_flops

    let y: f64 = 1.0;
    let _ = y * 2.0 + 0.5;
    //~^ suboptimal_flops
    let _ = 1.0 * 2.0 + 0.5;
    //~^ suboptimal_flops
}
