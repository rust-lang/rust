use autodiff::autodiff;

#[autodiff(d_sin, Forward, Active)]
fn sin1(x: f32) -> f32;

#[autodiff(d_sin, Reverse, Duplicated)]
fn sin2(x: f32) -> f32;

#[autodiff(d_sin, Reverse, DuplicatedNoNeed)]
fn sin3(x: f32) -> f32;

fn main() {}
