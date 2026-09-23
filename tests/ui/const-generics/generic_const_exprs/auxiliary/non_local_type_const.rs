#![feature(min_generic_const_args)]
#![allow(incomplete_features)]

use std::gca;

pub const NON_LOCAL_CONST: char = gca!('a');
