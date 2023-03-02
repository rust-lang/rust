use autodiff::autodiff;

#[autodiff(d_sin, Forward, Duplicated)]
fn fwd_output_no_reference(#[dup] x: &mut f32, y: f32) -> f32;

#[autodiff(d_sin, Forward, Duplicated)]
fn output_immutable(#[dup] x: &mut f32, y: &f32) -> f32;

#[autodiff(d_sin, Reverse, Active)]
fn rev_input_no_reference(#[dup] x: &f32, y: f32) -> f32;

#[autodiff(d_sin, Reverse, Active)]
fn rev_output_no_reference(#[dup] x: &mut f32, y: f32) -> f32;

#[autodiff(d_sin, Reverse, Active)]
fn input_immutable(#[dup] x: &f32, y: &f32) -> f32;

#[autodiff(d_sin, Reverse, Active)]
fn output_mutable(#[dup] x: &mut f32, y: &mut f32) -> f32;

#[autodiff(d_sin, Reverse, Active)]
fn dupnoneed_input(#[dup_noneed] x: &f32, y: &f32) -> f32;

fn main() {}
