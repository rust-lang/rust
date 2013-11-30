#[crate_type = "dylib"];
extern mod both;

use std::cast;

pub fn addr() -> uint { unsafe { cast::transmute(&both::foo) } }
