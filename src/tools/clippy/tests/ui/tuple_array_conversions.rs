//@aux-build:proc_macros.rs
#![allow(clippy::no_effect, clippy::useless_vec, unused)]
#![warn(clippy::tuple_array_conversions)]

#[macro_use]
extern crate proc_macros;

fn main() {
    let x = [1, 2];
    let x = (x[0], x[1]);
    //~^ tuple_array_conversions
    let x = [x.0, x.1];
    //~^ tuple_array_conversions
    let x = &[1, 2];
    let x = (x[0], x[1]);

    let t1: &[(u32, u32)] = &[(1, 2), (3, 4)];
    let v1: Vec<[u32; 2]> = t1.iter().map(|&(a, b)| [a, b]).collect();
    //~^ tuple_array_conversions
    t1.iter().for_each(|&(a, b)| _ = [a, b]);
    //~^ tuple_array_conversions
    let t2: Vec<(u32, u32)> = v1.iter().map(|&[a, b]| (a, b)).collect();
    //~^ tuple_array_conversions
    t1.iter().for_each(|&(a, b)| _ = [a, b]);
    //~^ tuple_array_conversions
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
    // FP #11082; needs discussion
    let (a, b) = (1.0f64, 2.0f64);
    let _: &[f64] = &[a, b];
    //~^ tuple_array_conversions
    // FP #11085; impossible to fix
    let [src, dest]: [_; 2] = [1, 2];
    (src, dest);
    //~^ tuple_array_conversions
    // FP #11100
    fn issue_11100_array_to_tuple(this: [&mut i32; 2]) -> (&i32, &mut i32) {
        let [input, output] = this;
        (input, output)
    }

    fn issue_11100_tuple_to_array<'a>(this: (&'a mut i32, &'a mut i32)) -> [&'a i32; 2] {
        let (input, output) = this;
        [input, output]
    }
    // FP #11124
    // tuple=>array
    let (a, b) = (1, 2);
    [a, b];
    let x = a;
    // array=>tuple
    let [a, b] = [1, 2];
    (a, b);
    let x = a;
    // FP #11144
    let (a, (b, c)) = (1, (2, 3));
    [a, c];
    let [[a, b], [c, d]] = [[1, 2], [3, 4]];
    (a, c);
    // Array length is not usize (#11144)
    fn generic_array_length<const N: usize>() {
        let src = [0; N];
        let dest: (u8,) = (src[0],);
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
    //~^ tuple_array_conversions
    let x = [x.0, x.1];
    //~^ tuple_array_conversions
    let x = &[1, 2];
    let x = (x[0], x[1]);
}
