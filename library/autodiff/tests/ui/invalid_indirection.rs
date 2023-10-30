use autodiff::autodiff;

#[autodiff(d_sin, Reverse, Const)]
fn duplicated_without_reference(#[dup] x: f32) {
}

#[autodiff(d_sin, Reverse, Const)]
fn active_with_reference(#[active] x: &f32) {
}

#[autodiff(d_sin, Forward, Const)]
fn duplicated_forward(#[dup] x: f32) {
}

#[autodiff(d_sin, Forward, Const)]
fn duplicated_no_need_forward(#[dup_noneed] x: &f32) {
}

fn main() {}
