//@edition:2018

#![warn(clippy::needless_parens_on_range_literals)]
#![allow(clippy::almost_complete_range)]

fn main() {
    let _ = ('a')..=('z');
    let _ = 'a'..('z');
    let _ = (1.)..2.;
    let _ = (1.)..(2.);
    let _ = ('a')..;
    let _ = ..('z');
}
