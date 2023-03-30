use autodiff::autodiff;

#[autodiff(d_sin, WrongMode)]
fn sin(x: f32) -> f32;

fn main() {}
