#[warn(clippy::approx_constant)]
fn main() {
    let my_e = 2.7182;
    //~^ approx_constant

    let almost_e = 2.718;
    //~^ approx_constant

    let no_e = 2.71;

    let my_1_frac_pi = 0.3183;
    //~^ approx_constant

    let no_1_frac_pi = 0.31;

    let my_frac_1_sqrt_2 = 0.70710678;
    //~^ approx_constant

    let almost_frac_1_sqrt_2 = 0.70711;
    //~^ approx_constant

    let my_frac_1_sqrt_2 = 0.707;

    let my_frac_2_pi = 0.63661977;
    //~^ approx_constant

    let no_frac_2_pi = 0.636;

    let my_frac_2_sq_pi = 1.128379;
    //~^ approx_constant

    let no_frac_2_sq_pi = 1.128;

    let my_frac_pi_2 = 1.57079632679;
    //~^ approx_constant

    let no_frac_pi_2 = 1.5705;

    let my_frac_pi_3 = 1.04719755119;
    //~^ approx_constant

    let no_frac_pi_3 = 1.047;

    let my_frac_pi_4 = 0.785398163397;
    //~^ approx_constant

    let no_frac_pi_4 = 0.785;

    let my_frac_pi_6 = 0.523598775598;
    //~^ approx_constant

    let no_frac_pi_6 = 0.523;

    let my_frac_pi_8 = 0.3926990816987;
    //~^ approx_constant

    let no_frac_pi_8 = 0.392;

    let my_ln_10 = 2.302585092994046;
    //~^ approx_constant

    let no_ln_10 = 2.303;

    let my_ln_2 = 0.6931471805599453;
    //~^ approx_constant

    let no_ln_2 = 0.693;

    let my_log10_e = 0.4342944819032518;
    //~^ approx_constant

    let no_log10_e = 0.434;

    let my_log2_e = 1.4426950408889634;
    //~^ approx_constant

    let no_log2_e = 1.442;

    let log2_10 = 3.321928094887362;
    //~^ approx_constant

    let no_log2_10 = 3.321;

    let log10_2 = 0.301029995663981;
    //~^ approx_constant

    let no_log10_2 = 0.301;

    let my_pi = 3.1415;
    //~^ approx_constant

    let almost_pi = 3.14;
    //~^ approx_constant

    let no_pi = 3.15;

    let my_sq2 = 1.4142;
    //~^ approx_constant

    let no_sq2 = 1.414;

    let my_tau = 6.2832;
    //~^ approx_constant

    let almost_tau = 6.28;
    //~^ approx_constant

    let no_tau = 6.3;

    // issue #15194
    #[allow(clippy::excessive_precision)]
    let x: f64 = 3.1415926535897932384626433832;
    //~^ approx_constant

    #[allow(clippy::excessive_precision)]
    let _: f64 = 003.14159265358979311599796346854418516159057617187500;
    //~^ approx_constant

    let almost_frac_1_sqrt_2 = 00.70711;
    //~^ approx_constant

    let almost_frac_1_sqrt_2 = 00.707_11;
    //~^ approx_constant
}
