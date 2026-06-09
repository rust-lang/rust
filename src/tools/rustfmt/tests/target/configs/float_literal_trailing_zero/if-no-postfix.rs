// rustfmt-float_literal_trailing_zero: IfNoPostfix

fn float_literals() {
    let a = 0.0;
    let b = 0.0;
    let c = 100.0;
    let d = 100.0;
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
    let q = 1000.00;
    let r = 1_000_.0;
    let s = 1_000_.000_000;
}

fn range_bounds() {
    if (1.0..2.0).contains(&1.0) {}
    if (1.1..2.2).contains(&1.1) {}
    if (1e1..2e1).contains(&1e1) {}
    let _binop_range = 3.0 / 2.0..4.0;
}

fn method_calls() {
    let x = 1.0.neg();
    let y = 2.3.neg();
    let z = (4.0).neg();
    let u = 5f32.neg();
    let v = -6.0.neg();
}

fn line_wrapping() {
    let array = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0,
    ];
    println!("This is floaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaat {}", 10e3);
}
