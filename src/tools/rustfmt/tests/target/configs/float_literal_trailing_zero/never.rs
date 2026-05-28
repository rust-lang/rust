// rustfmt-float_literal_trailing_zero: Never

fn float_literals() {
    let a = 0.;
    let b = 0.;
    let c = 100.;
    let d = 100.;
    let e = 5e3;
    let f = 5e3;
    let g = 5e+3;
    let h = 5e+3;
    let i = 5e-3;
    let j = 5e-3;
    let k = 5E3;
    let l = 5E3;
    let m = 7f32;
    let n = 7f32;
    let o = 9e3f32;
    let p = 9e3f32;
    let q = 1000.;
    let r = 1_000_.;
    let s = 1_000_.;
}

fn range_bounds() {
    if (1. ..2.).contains(&1.) {}
    if (1.1..2.2).contains(&1.1) {}
    if (1e1..2e1).contains(&1e1) {}
    let _binop_range = 3. / 2. ..4.;
}

fn method_calls() {
    let x = (1.).neg();
    let y = 2.3.neg();
    let z = (4.).neg();
    let u = 5f32.neg();
    let v = -(6.).neg();
}

fn line_wrapping() {
    let array = [
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
    ];
    println!("This is floaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaat {}", 10e3);
}
