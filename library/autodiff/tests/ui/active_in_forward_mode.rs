use autodiff::autodiff;

#[autodiff(d_sin, Forward, DuplicatedNoNeed, Active)]
fn sin(x: f32) -> f32;

fn main() {}
