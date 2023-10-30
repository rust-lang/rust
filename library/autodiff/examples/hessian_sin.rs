use autodiff::autodiff;

fn sin(x: &Vec<f32>, y: &mut f32) {
    *y = x.into_iter().map(|x| f32::sin(*x)).sum()
}

#[autodiff(sin, Reverse, Const, Duplicated, Duplicated)]
fn jac(x: &Vec<f32>, d_x: &mut Vec<f32>, y: &mut f32, y_t: &f32);

#[autodiff(jac, Forward, Const, Duplicated, Const, Const, Const)]
fn hessian(x: &Vec<f32>, y_x: &Vec<f32>, d_x: &mut Vec<f32>, y: &mut f32, y_t: &f32);

fn main() {
    let inp = vec![3.1415 / 2., 1.0, 0.5];
    let mut d_inp = vec![0.0, 0.0, 0.0];
    let mut y = 0.0;
    let tang = vec![1.0, 0.0, 0.0];
    hessian(&inp, &tang, &mut d_inp, &mut y, &1.0);
    dbg!(&d_inp);
}

#[cfg(test)]
mod tests {
    #[test]
    fn main() {
        super::main()
    }
}
