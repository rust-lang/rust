#[allow(unused_variables)]
#[warn(clippy::zero_divided_by_zero)]
fn main() {
    let nan = 0.0 / 0.0;
    let f64_nan = 0.0 / 0.0f64;
    let other_f64_nan = 0.0f64 / 0.0;
    let one_more_f64_nan = 0.0f64 / 0.0f64;
    let zero = 0.0;
    let other_zero = 0.0;
    let other_nan = zero / other_zero; // fine - this lint doesn't propegate constants.
    let not_nan = 2.0 / 0.0; // not an error: 2/0 = inf
    let also_not_nan = 0.0 / 2.0; // not an error: 0/2 = 0
}
