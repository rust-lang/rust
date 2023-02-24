#![feature(bench_black_box)]
use autodiff::autodiff;

fn sin(x: &f32) -> f32 {
    f32::sin(*x)
}

#[autodiff(sin, Reverse, Active, Active)]
fn cos(x: &f32, adj: f32) -> f32;

//#[autodiff(cos, Reverse, Active, Active, Const)]
//fn neg_sin(x: &f32, adj: f32, adj_sec: f32) -> f32;

fn main() {
    dbg!(&cos(&1.0, 1.0));
    //dbg!(&neg_sin(&1.0, 1.0, 1.0));
}
