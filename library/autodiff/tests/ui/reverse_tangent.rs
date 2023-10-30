use autodiff::autodiff;

#[autodiff(d_sin, Reverse, Active)]
fn invalid_output_tangent_type(#[active] x: f32, y_tang: i32) -> f32;

#[autodiff(d_sin, Reverse, Active)]
fn active_output_tangent(#[active] x: f32, #[active] y_tang: f32) -> f32;

#[autodiff(d_sin, Reverse, Active)]
fn tangent_missing() -> f32;

fn main() {}
