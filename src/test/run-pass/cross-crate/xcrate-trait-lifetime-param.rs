// run-pass
#![allow(dead_code)]
// aux-build:xcrate-trait-lifetime-param.rs

// pretty-expanded FIXME #23616

extern crate xcrate_trait_lifetime_param as other;

struct Reader<'a> {
    b : &'a [u8]
}

impl <'a> other::FromBuf<'a> for Reader<'a> {
    fn from_buf(b : &'a [u8]) -> Reader<'a> {
        Reader { b : b }
    }
}

pub fn main () {}
