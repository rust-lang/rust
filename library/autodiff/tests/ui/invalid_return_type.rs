use autodiff::autodiff;

#[autodiff(d_sin, Reverse, Active)]
fn active_but_no_return(#[active] x: f32) {
}

#[autodiff(d_sin, Reverse, Active)]
fn invalid_primal_value(#[active] x: f32, #[active] y: Vec<f32>, #[active] z: Tensor, y_tang: f32) -> (i32, f32);

#[autodiff(d_sin, Forward, Duplicated)]
fn invalid_forward_return(#[dup] x: &f32, tx: &f32, #[dup] y: &Vec<f32>, ty: &Vec<f32>, #[dup] z: &Tensor, tz: &Tensor) -> (f32, f32, f32);

#[autodiff(d_sin, Forward, DuplicatedNoNeed)]
fn invalid_forward_return(#[dup] x: &f32, tx: &f32, #[dup] y: &Vec<f32>, ty: &Vec<f32>, #[dup] z: &Tensor, tz: &Tensor) -> (f32, f32);

fn main() {}
