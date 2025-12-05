//@edition:2018

#![warn(clippy::needless_parens_on_range_literals)]
#![allow(clippy::almost_complete_range)]

fn main() {
    let _ = ('a')..=('z');
    //~^ needless_parens_on_range_literals
    //~| needless_parens_on_range_literals
    let _ = 'a'..('z');
    //~^ needless_parens_on_range_literals
    let _ = (1.)..2.;
    let _ = (1.)..(2.);
    //~^ needless_parens_on_range_literals
    let _ = ('a')..;
    //~^ needless_parens_on_range_literals
    let _ = ..('z');
    //~^ needless_parens_on_range_literals
}
