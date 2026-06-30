#![feature(autodiff)]
use std::autodiff::*;
use std::f64::consts::FRAC_PI_6;
use ndarray::arr1;

#[autodiff_forward(da_args, Dual, Dual)]
pub fn a_f64_args(args: &[f64]) -> f64 {
    let temperature = args[0];
    let volume = args[1];
    let moles = &args[2..];

    // Now this crashes as well ...
    let m = arr1(&[2.001829]);
    let sigma = arr1(&[3.618353]);
    let epsilon_k = arr1(&[208.1101]);
    
    // ... as does this ...
    //let m = vec![2.001829];
    //let sigma = vec![3.618353];
    //let epsilon_k = vec![208.1101];

    // but this works
    // let m = &[2.001829];
    // let sigma = &[3.618353];
    // let epsilon_k = &[208.1101];

    let n = moles.len();
    let t_inv = temperature.recip();
    let diameter: Vec<f64> = (0..n)
        .map(|i| -((t_inv * -3.0 * epsilon_k[i]).exp() * 0.12 - 1.0) * sigma[i])
        .collect();

    //let partial_density: Vec<f64> = moles.iter().cloned().map(|n| n / volume).collect(); // <- works
    let partial_density: Vec<f64> = moles.iter().map(|&n| n / volume).collect(); // <- crashes
    let density: f64 = partial_density.iter().cloned().sum();
    let total_moles: f64 = moles.iter().cloned().sum();
    let x: Vec<f64> = moles.iter().cloned().map(|n| n / total_moles).collect();

    let mut zeta = [0.0; 4];
    for i in 0..diameter.len() {
        for (z, &k) in zeta.iter_mut().zip([0, 1, 2, 3].iter()) {
            *z += x[i] * diameter[i].powi(k) * (m[i] * FRAC_PI_6);
        }
    }
    let zeta_23 = zeta[2] / zeta[3];

    zeta.iter_mut().for_each(|z| *z *= density);
    let frac_1mz3 = -(zeta[3] - 1.0).recip();
    let a = volume / std::f64::consts::FRAC_PI_6
        * (zeta[1] * zeta[2] * frac_1mz3 * 3.0
            + zeta[2].powi(2) * frac_1mz3.powi(2) * zeta_23
            + (zeta[2] * zeta_23.powi(2) - zeta[0]) * (zeta[3] * (-1.0)).ln_1p());
    a
}

fn main() {
    let t = 250.0;
    let v = 1000.0;
        
    let seed = &[1.0, 0.0, 0.0]; // first entry is temperature
    let da_dt_enz_args = da_args(&[t, v, 1.0], seed);
    dbg!(&da_dt_enz_args);
}
