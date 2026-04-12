// test that layout cycles involving Align error and don't ICE
#![feature(align_type)]

use std::mem::Align;

struct Evil {
    align: Align<{align_of::<Evil>()}>, //~ ERROR cycle detected
}

fn main() {}
