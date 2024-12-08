//@ run-pass
//@ edition: 2021
//@ revisions: classic structural both
#![allow(incomplete_features)]
#![cfg_attr(any(classic, both), feature(ref_pat_eat_one_layer_2024))]
#![cfg_attr(any(structural, both), feature(ref_pat_eat_one_layer_2024_structural))]

pub fn main() {
    if let &Some(Some(x)) = &Some(&mut Some(0)) {
        let _: &u32 = x;
    }
    if let Some(&x) = Some(&mut 0) {
        let _: u32 = x;
    }
}
