//@ check-pass
//! See also: tests/ui/const-generics/paren.rs
#![feature(gca_min_const_items, generic_const_items)]

use std::gca;

struct Thing<const N: usize>;

const A<const N: usize>: usize = gca!(N);

fn f<const N: usize>() {
    let _: [u32; gca!(_)] = [5; 5];
    let _: [u32; gca!((_))] = [5; 5];
    let _: [u32; gca!({ _ })] = [5; 5];
    let _: [u32; gca!({ (_) })] = [5; 5];
    let _: [u32; gca!(N)] = [5; _];
    let _: [u32; gca!((N))] = [5; _];
    let _: [u32; gca!({ N })] = [5; _];
    let _: [u32; gca!({ (N) })] = [5; _];
    let _: [u32; gca!(A::<N>)] = [5; _];
    let _: [u32; gca!((A::<N>))] = [5; _];
    let _: [u32; gca!({ A::<N> })] = [5; _];
    let _: [u32; gca!({ (A::<N>) })] = [5; _];
    let _: Thing<gca!(_)> = Thing::<5>;
    let _: Thing<gca!((_))> = Thing::<5>;
    let _: Thing<gca!({ _ })> = Thing::<5>;
    let _: Thing<gca!({ (_) })> = Thing::<5>;
    let _: Thing<gca!(N)> = Thing;
    let _: Thing<gca!((N))> = Thing;
    let _: Thing<gca!({ N })> = Thing;
    let _: Thing<gca!({ (N) })> = Thing;
    let _: Thing<gca!(A::<N>)> = Thing;
    let _: Thing<gca!((A::<N>))> = Thing;
    let _: Thing<gca!({ A::<N> })> = Thing;
    let _: Thing<gca!({ (A::<N>) })> = Thing;
}

fn main() {}
