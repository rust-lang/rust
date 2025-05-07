//@ run-pass
#![allow(unused_imports, overlapping_range_endpoints)]

use m::{START, END};

fn main() {
    match 42 {
        m::START..=m::END => {},
        0..=m::END => {},
        m::START..=59 => {},
        _  => {},
    }
}

mod m {
  pub const START: u32 = 4;
  pub const END:   u32 = 14;
}
