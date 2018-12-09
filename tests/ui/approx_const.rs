// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[warn(clippy::approx_constant)]
#[allow(unused, clippy::shadow_unrelated, clippy::similar_names, clippy::unreadable_literal)]
fn main() {
    let my_e = 2.7182;
    let almost_e = 2.718;
    let no_e = 2.71;

    let my_1_frac_pi = 0.3183;
    let no_1_frac_pi = 0.31;

    let my_frac_1_sqrt_2 = 0.70710678;
    let almost_frac_1_sqrt_2 = 0.70711;
    let my_frac_1_sqrt_2 = 0.707;

    let my_frac_2_pi = 0.63661977;
    let no_frac_2_pi = 0.636;

    let my_frac_2_sq_pi = 1.128379;
    let no_frac_2_sq_pi = 1.128;

    let my_frac_pi_2 = 1.57079632679;
    let no_frac_pi_2 = 1.5705;

    let my_frac_pi_3 = 1.04719755119;
    let no_frac_pi_3 = 1.047;

    let my_frac_pi_4 = 0.785398163397;
    let no_frac_pi_4 = 0.785;

    let my_frac_pi_6 = 0.523598775598;
    let no_frac_pi_6 = 0.523;

    let my_frac_pi_8 = 0.3926990816987;
    let no_frac_pi_8 = 0.392;

    let my_ln_10 = 2.302585092994046;
    let no_ln_10 = 2.303;

    let my_ln_2 = 0.6931471805599453;
    let no_ln_2 = 0.693;

    let my_log10_e = 0.4342944819032518;
    let no_log10_e = 0.434;

    let my_log2_e = 1.4426950408889634;
    let no_log2_e = 1.442;

    let my_pi = 3.1415;
    let almost_pi = 3.14;
    let no_pi = 3.15;

    let my_sq2 = 1.4142;
    let no_sq2 = 1.414;
}
