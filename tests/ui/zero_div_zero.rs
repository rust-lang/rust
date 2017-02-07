#![feature(plugin)]
#![plugin(clippy)]

#[allow(unused_variables)]
#[deny(zero_divided_by_zero)]
fn main() {
    let nan = 0.0 / 0.0; //~ERROR constant division of 0.0 with 0.0 will always result in NaN
                         //~^ equal expressions as operands to `/`
    let f64_nan = 0.0 / 0.0f64; //~ERROR constant division of 0.0 with 0.0 will always result in NaN
                         //~^ equal expressions as operands to `/`
    let other_f64_nan = 0.0f64 / 0.0; //~ERROR constant division of 0.0 with 0.0 will always result in NaN
                         //~^ equal expressions as operands to `/`
    let one_more_f64_nan = 0.0f64/0.0f64; //~ERROR constant division of 0.0 with 0.0 will always result in NaN
                         //~^ equal expressions as operands to `/`
    let zero = 0.0;
    let other_zero = 0.0;
    let other_nan = zero / other_zero; // fine - this lint doesn't propegate constants.
    let not_nan = 2.0/0.0; // not an error: 2/0 = inf
    let also_not_nan = 0.0/2.0; // not an error: 0/2 = 0
}
