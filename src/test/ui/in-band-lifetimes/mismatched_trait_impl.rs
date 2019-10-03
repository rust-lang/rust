#![allow(warnings)]
#![feature(in_band_lifetimes)]

trait Get {
    fn foo(&self, x: &'a u32, y: &u32) -> &'a u32;
}

impl Get for i32 {
    fn foo(&self, x: &u32, y: &'a u32) -> &'a u32 { //~ ERROR `impl` item doesn't match `trait`
        x //~ ERROR lifetime mismatch
    }
}

fn main() {}
