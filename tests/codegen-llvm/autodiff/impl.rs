//@ compile-flags: -Zautodiff=Enable -Zautodiff=NoPostopt -C opt-level=3 -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme

// Just check it does not crash for now
// CHECK: ;
#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[derive(Clone)]
struct OptProblem {
    a: f64,
    b: f64,
}

impl OptProblem {
    #[autodiff_reverse(d_objective, Duplicated, Duplicated, Duplicated)]
    fn objective(&self, x: &[f64], out: &mut f64) {
        *out = self.a + x[0].sqrt() * self.b
    }
}

fn main() {
    let p = OptProblem { a: 1., b: 2. };
    let x = [2.0];

    let mut p_shadow = OptProblem { a: 0., b: 0. };
    let mut dx = [0.0];
    let mut out = 0.0;
    let mut dout = 1.0;

    p.d_objective(&mut p_shadow, &x, &mut dx, &mut out, &mut dout);

    dbg!(dx);
}
