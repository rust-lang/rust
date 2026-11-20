#![crate_type="lib"]
#![feature(move_trait)]
#![feature(negative_impls)]
#![feature(min_specialization)]

use std::marker::Move;

pub struct NotMove;
impl !Move for NotMove {}

pub fn requires_move<T: Move>(m: T) {}
