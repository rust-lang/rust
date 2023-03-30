use autodiff::autodiff;

#[autodiff(d_sin, Reverse, Active, Active)]
fn sin(#[active] x: f32) -> f32;

fn main() {}
