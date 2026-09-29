#![feature(gca_min_const_items)]
#![allow(incomplete_features)]

use std::gca;

pub const NON_LOCAL_CONST: char = gca!('a');
