#![feature(move_trait, negative_impls, min_specialization)]
#![allow(incomplete_features)]

use std::marker::Move;

pub struct NotMove;

impl !Move for NotMove {}

pub fn need<T: Move>(_: T) {}
