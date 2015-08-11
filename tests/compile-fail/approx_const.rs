#![feature(plugin)]
#![plugin(clippy)]

#[deny(approx_constant)]
#[allow(unused)]
fn main() {
    let my_e = 2.7182; //~ERROR
    let almost_e = 2.718; //~ERROR
    let no_e = 2.71;

    let my_1_frac_pi = 0.3183; //~ERROR
    let no_1_frac_pi = 0.31;

    let my_frac_1_sqrt_2 = 0.70710678; //~ERROR
    let almost_frac_1_sqrt_2 = 0.70711; //~ERROR
    let my_frac_1_sqrt_2 = 0.707;

    let my_frac_2_pi = 0.63661977; //~ERROR
    let no_frac_2_pi = 0.636;

    let my_frac_2_sq_pi = 1.128379; //~ERROR
    let no_frac_2_sq_pi = 1.128;

    let my_frac_2_pi = 1.57079632679; //~ERROR
    let no_frac_2_pi = 1.5705;

    let my_frac_3_pi = 1.04719755119; //~ERROR
    let no_frac_3_pi = 1.047;

    let my_frac_4_pi = 0.785398163397; //~ERROR
    let no_frac_4_pi = 0.785;

    let my_frac_6_pi = 0.523598775598; //~ERROR
    let no_frac_6_pi = 0.523;

    let my_frac_8_pi = 0.3926990816987; //~ERROR
    let no_frac_8_pi = 0.392;

    let my_ln_10 = 2.302585092994046; //~ERROR
    let no_ln_10 = 2.303;

    let my_ln_2 = 0.6931471805599453; //~ERROR
    let no_ln_2 = 0.693;

    let my_log10_e = 0.43429448190325176; //~ERROR
    let no_log10_e = 0.434;

    let my_log2_e = 1.4426950408889634; //~ERROR
    let no_log2_e = 1.442;

    let my_pi = 3.1415; //~ERROR
    let almost_pi = 3.141;

    let my_sq2 = 1.4142; //~ERROR
    let no_sq2 = 1.414;
}
