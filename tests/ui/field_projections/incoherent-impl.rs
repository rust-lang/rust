//@ aux-build:extern-crate.rs
#![allow(incomplete_features)]
#![feature(field_projections)]
extern crate extern_crate;

use std::field::field_of;

use extern_crate::{ForeignTrait, Point};

pub trait MyTrait {}

impl MyTrait for field_of!(Point, x) {}

impl extern_crate::ForeignTrait for field_of!(Point, x) {}
//~^ ERROR: only traits defined in the current crate can be implemented for arbitrary types [E0117]

pub struct Player {
    pos: Point,
    hp: (u32, u32),
}

impl ForeignTrait for field_of!(Player, pos) {}

impl ForeignTrait for field_of!(Player, pos.x) {}
//~^ ERROR: only traits defined in the current crate can be implemented for arbitrary types [E0117]

impl MyTrait for field_of!(Player, pos) {}

impl MyTrait for field_of!(Player, pos.x) {}

impl MyTrait for field_of!((usize, usize), 0) {}

impl ForeignTrait for field_of!((usize, usize), 0) {}
//~^ ERROR: only traits defined in the current crate can be implemented for arbitrary types [E0117]

impl ForeignTrait for field_of!(Player, hp.0) {}

fn main() {}
