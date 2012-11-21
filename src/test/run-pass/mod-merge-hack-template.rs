// xfail-test not a test. used by mod-merge-hack.rs

use T = inst::T;

pub const bits: uint = inst::bits;
pub pure fn min(x: T, y: T) -> T { if x < y { x } else { y } }
