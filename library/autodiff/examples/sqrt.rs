#![feature(bench_black_box)]

#[autodiff_into]
fn sqrt(a: f32, b: &f32, c: f32, d: f32) -> f32 {
    a * (b * b + c * c * d * d).sqrt()
}
#[autodiff_into(Reverse, Active, Active, Duplicated, Const, Active)]
fn d_sqrt(a: f32, b: &f32, d_b: &mut f32, c: f32, d: f32, ret_t: f32) -> (f32, f32) {
    std::hint::black_box((sqrt(a, b, c, d), &d_b, &ret_t, &a, &b, &c, &d));
    unsafe { std::mem::zeroed() }
}
//#[autodiff(d_sqrt, Reverse, Active)]
//fn sqrt(#[active] a: &f32, #[dup] b: &f32, c: &f32, #[active] d: &f32) -> f32 {
//    a * (b * b + c*c*d*d).sqrt()
//}

fn main() {
    let mut d_b = 0.0;

    let (d_a, d_d) = d_sqrt(1.0, &1.0, &mut d_b, 1.0, 1.0, 1.0);

    dbg!(d_a, d_b, d_d);
}
