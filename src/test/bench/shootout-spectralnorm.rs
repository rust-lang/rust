use std::f64;
use std::from_str::FromStr;
use std::iter::ExtendedMutableIter;
use std::os;
use std::vec;

#[inline]
fn A(i: i32, j: i32) -> i32 {
    (i+j) * (i+j+1) / 2 + i + 1
}

fn dot(v: &[f64], u: &[f64]) -> f64 {
    let mut sum = 0.0;
    for v.eachi |i, &v_i| {
        sum += v_i * u[i];
    }
    sum
}

fn mult_Av(v: &mut [f64], out: &mut [f64]) {
    for vec::eachi_mut(out) |i, out_i| {
        let mut sum = 0.0;
        for vec::eachi_mut(v) |j, &v_j| {
            sum += v_j / (A(i as i32, j as i32) as f64);
        }
        *out_i = sum;
    }
}

fn mult_Atv(v: &mut [f64], out: &mut [f64]) {
    for vec::eachi_mut(out) |i, out_i| {
        let mut sum = 0.0;
        for vec::eachi_mut(v) |j, &v_j| {
            sum += v_j / (A(j as i32, i as i32) as f64);
        }
        *out_i = sum;
    }
}

fn mult_AtAv(v: &mut [f64], out: &mut [f64], tmp: &mut [f64]) {
    mult_Av(v, tmp);
    mult_Atv(tmp, out);
}

#[fixed_stack_segment]
fn main() {
    let n: uint = FromStr::from_str(os::args()[1]).get();
    let mut u = vec::from_elem(n, 1f64);
    let mut v = u.clone();
    let mut tmp = u.clone();
    for 8.times {
        mult_AtAv(u, v, tmp);
        mult_AtAv(v, u, tmp);
    }

    println(fmt!("%.9f", f64::sqrt(dot(u,v) / dot(v,v)) as float));
}
