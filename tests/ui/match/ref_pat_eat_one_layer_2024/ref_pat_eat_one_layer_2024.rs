//@ run-pass
//@ edition: 2024
//@ compile-flags: -Zunstable-options
#![allow(incomplete_features)]
#![feature(ref_pat_eat_one_layer_2024)]

pub fn main() {
    if let Some(Some(&x)) = &Some(&Some(0)) {
        let _: u32 = x;
    }
    if let Some(Some(&x)) = &Some(Some(&0)) {
        let _: &u32 = x;
    }
    if let Some(Some(&&x)) = &Some(Some(&0)) {
        let _: u32 = x;
    }
    if let Some(&Some(x)) = &Some(Some(0)) {
        let _: u32 = x;
    }
    if let Some(Some(&mut x)) = &mut Some(&mut Some(0)) {
        let _: u32 = x;
    }
    if let Some(Some(&x)) = &Some(&Some(0)) {
        let _: u32 = x;
    }
    if let Some(&Some(&x)) = &mut Some(&Some(0)) {
        let _: u32 = x;
    }
    if let Some(&Some(x)) = &mut Some(&Some(0)) {
        let _: &u32 = x;
    }
    if let Some(&Some(&mut ref x)) = Some(&Some(&mut 0)) {
        let _: &u32 = x;
    }
    if let &Some(Some(x)) = &Some(&mut Some(0)) {
        let _: &u32 = x;
    }
    if let Some(&Some(&x)) = &Some(&mut Some(0)) {
        let _: u32 = x;
    }
    if let Some(&Some(&x)) = &Some(&Some(0)) {
        let _: u32 = x;
    }
    if let Some(&Some(&x)) = &Some(&mut Some(0)) {
        let _: u32 = x;
    }
    if let Some(&Some(Some(&x))) = &Some(Some(&mut Some(0))) {
        let _: u32 = x;
    }
    if let Some(&Some(&x)) = Some(&Some(&mut 0)) {
        let _: u32 = x;
    }
    if let Some(&Some(x)) = &mut Some(Some(0)) {
        let _: u32 = x;
    }
}
