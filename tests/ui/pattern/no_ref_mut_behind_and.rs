//@ edition: 2021
//@ run-pass
#![allow(incomplete_features)]
#![feature(ref_pat_eat_one_layer_2024)]

fn main() {
    let &[[x]] = &[&mut [42]];
    let _: &i32 = x;
}
