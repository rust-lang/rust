//@aux-build:proc_macros.rs:proc-macro
#![allow(clippy::no_effect, clippy::useless_vec, unused)]
#![warn(clippy::tuple_array_conversions)]

#[macro_use]
extern crate proc_macros;

fn main() {
    let x = [1, 2];
    let x = (x[0], x[1]);
    let x = [x.0, x.1];
    let x = &[1, 2];
    let x = (x[0], x[1]);

    let t1: &[(u32, u32)] = &[(1, 2), (3, 4)];
    let v1: Vec<[u32; 2]> = t1.iter().map(|&(a, b)| [a, b]).collect();
    t1.iter().for_each(|&(a, b)| _ = [a, b]);
    let t2: Vec<(u32, u32)> = v1.iter().map(|&[a, b]| (a, b)).collect();
    t1.iter().for_each(|&(a, b)| _ = [a, b]);
    // Do not lint
    let v2: Vec<[u32; 2]> = t1.iter().map(|&t| t.into()).collect();
    let t3: Vec<(u32, u32)> = v2.iter().map(|&v| v.into()).collect();
    let x = [1; 13];
    let x = (
        x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12],
    );
    let x = [x.0, x.1, x.2, x.3, x.4, x.5, x.6, x.7, x.8, x.9, x.10, x.11, x.12];
    let x = (1, 2);
    let x = (x.0, x.1);
    let x = [1, 2];
    let x = [x[0], x[1]];
    let x = vec![1, 2];
    let x = (x[0], x[1]);
    let x = [1; 3];
    let x = (x[0],);
    let x = (1, 2, 3);
    let x = [x.0];
    let x = (1, 2);
    let y = (1, 2);
    [x.0, y.0];
    [x.0, y.1];
    let x = [x.0, x.0];
    let x = (x[0], x[0]);
    external! {
        let t1: &[(u32, u32)] = &[(1, 2), (3, 4)];
        let v1: Vec<[u32; 2]> = t1.iter().map(|&(a, b)| [a, b]).collect();
        let t2: Vec<(u32, u32)> = v1.iter().map(|&[a, b]| (a, b)).collect();
    }
    with_span! {
        span
        let t1: &[(u32, u32)] = &[(1, 2), (3, 4)];
        let v1: Vec<[u32; 2]> = t1.iter().map(|&(a, b)| [a, b]).collect();
        let t2: Vec<(u32, u32)> = v1.iter().map(|&[a, b]| (a, b)).collect();
    }
}

#[clippy::msrv = "1.70.0"]
fn msrv_too_low() {
    let x = [1, 2];
    let x = (x[0], x[1]);
    let x = [x.0, x.1];
    let x = &[1, 2];
    let x = (x[0], x[1]);
}

#[clippy::msrv = "1.71.0"]
fn msrv_juust_right() {
    let x = [1, 2];
    let x = (x[0], x[1]);
    let x = [x.0, x.1];
    let x = &[1, 2];
    let x = (x[0], x[1]);
}
