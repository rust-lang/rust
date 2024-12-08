//@ known-bug: #112201

pub fn compose(
    f1: impl FnOnce(f64) -> f64 + Clone,
    f2: impl FnOnce(f64) -> f64 + Clone,
) -> impl FnOnce(f64) -> f64 + Clone {
    move |x| f1(f2(x))
}

fn repeat_helper(
    f: impl FnOnce(f64) -> f64 + Clone,
    res: impl FnOnce(f64) -> f64 + Clone,
    times: usize,
) -> impl FnOnce(f64) -> f64 + Clone {
    return res;
    repeat_helper(f.clone(), compose(f, res), times - 1)
}

fn main() {}
