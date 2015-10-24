#![feature(plugin)]
#![plugin(clippy)]

#[deny(approx_constant)]
#[allow(unused, shadow_unrelated)]
fn main() {
    let my_e = 2.7182; //~ERROR approximate value of `f{32, 64}::E` found
    let almost_e = 2.718; //~ERROR approximate value of `f{32, 64}::E` found
    let no_e = 2.71;

    let my_1_frac_pi = 0.3183; //~ERROR approximate value of `f{32, 64}::FRAC_1_PI` found
    let no_1_frac_pi = 0.31;

    let my_frac_1_sqrt_2 = 0.70710678; //~ERROR approximate value of `f{32, 64}::FRAC_1_SQRT_2` found
    let almost_frac_1_sqrt_2 = 0.70711; //~ERROR approximate value of `f{32, 64}::FRAC_1_SQRT_2` found
    let my_frac_1_sqrt_2 = 0.707;

    let my_frac_2_pi = 0.63661977; //~ERROR approximate value of `f{32, 64}::FRAC_2_PI` found
    let no_frac_2_pi = 0.636;

    let my_frac_2_sq_pi = 1.128379; //~ERROR approximate value of `f{32, 64}::FRAC_2_SQRT_PI` found
    let no_frac_2_sq_pi = 1.128;

    let my_frac_pi_2 = 1.57079632679; //~ERROR approximate value of `f{32, 64}::FRAC_PI_2` found
    let no_frac_pi_2 = 1.5705;

    let my_frac_pi_3 = 1.04719755119; //~ERROR approximate value of `f{32, 64}::FRAC_PI_3` found
    let no_frac_pi_3 = 1.047;

    let my_frac_pi_4 = 0.785398163397; //~ERROR approximate value of `f{32, 64}::FRAC_PI_4` found
    let no_frac_pi_4 = 0.785;

    let my_frac_pi_6 = 0.523598775598; //~ERROR approximate value of `f{32, 64}::FRAC_PI_6` found
    let no_frac_pi_6 = 0.523;

    let my_frac_pi_8 = 0.3926990816987; //~ERROR approximate value of `f{32, 64}::FRAC_PI_8` found
    let no_frac_pi_8 = 0.392;

    let my_ln_10 = 2.302585092994046; //~ERROR approximate value of `f{32, 64}::LN_10` found
    let no_ln_10 = 2.303;

    let my_ln_2 = 0.6931471805599453; //~ERROR approximate value of `f{32, 64}::LN_2` found
    let no_ln_2 = 0.693;

    let my_log10_e = 0.43429448190325182; //~ERROR approximate value of `f{32, 64}::LOG10_E` found
    let no_log10_e = 0.434;

    let my_log2_e = 1.4426950408889634; //~ERROR approximate value of `f{32, 64}::LOG2_E` found
    let no_log2_e = 1.442;

    let my_pi = 3.1415; //~ERROR approximate value of `f{32, 64}::PI` found
    let almost_pi = 3.14; //~ERROR approximate value of `f{32, 64}::PI` found
    let no_pi = 3.15;

    let my_sq2 = 1.4142; //~ERROR approximate value of `f{32, 64}::SQRT_2` found
    let no_sq2 = 1.414;
}
