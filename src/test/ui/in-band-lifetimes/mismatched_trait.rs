#![allow(warnings)]
#![feature(in_band_lifetimes)]

trait Get {
    fn baz(&self, x: &'a u32, y: &u32) -> &'a u32 {
        y //~ ERROR explicit lifetime required
    }
}

fn main() {}
