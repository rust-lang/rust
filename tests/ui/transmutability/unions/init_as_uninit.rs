//@ check-pass
// Regression test for issue #140337.
#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code)]
use std::mem::{Assume, MaybeUninit, TransmuteFrom};

pub fn is_transmutable<Src, Dst>()
where
    Dst: TransmuteFrom<Src, { Assume::SAFETY }>
{}

#[derive(Copy, Clone)]
#[repr(u8)]
pub enum B0 { Value = 0 }

#[derive(Copy, Clone)]
#[repr(u8)]
pub enum B1 { Value = 1 }

fn main() {
    is_transmutable::<(B0, B0), MaybeUninit<(B0, B0)>>();
    is_transmutable::<(B0, B0), MaybeUninit<(B0, B1)>>();
    is_transmutable::<(B0, B0), MaybeUninit<(B1, B0)>>();
    is_transmutable::<(B0, B0), MaybeUninit<(B1, B1)>>();
}
