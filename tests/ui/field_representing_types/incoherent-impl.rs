//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
//@ aux-build:extern-crate.rs
#![expect(incomplete_features)]
#![feature(field_projections)]
extern crate extern_crate;

use std::field::field_of;

use extern_crate::{ForeignTrait, Point};

pub trait MyTrait {}

impl MyTrait for field_of!(Point, x) {}
impl MyTrait for field_of!(Player, pos) {}
impl MyTrait for field_of!((usize, usize), 0) {}

pub struct Player {
    pos: Point,
}

impl ForeignTrait for field_of!(Point, x) {}
//~^ ERROR: only traits defined in the current crate can be implemented for arbitrary types [E0117]
impl ForeignTrait for field_of!(Player, pos) {}
impl ForeignTrait for field_of!((usize, usize), 0) {}
//~^ ERROR: only traits defined in the current crate can be implemented for arbitrary types [E0117]

fn main() {}
