// pretty-expanded FIXME #23616

#![allow(dead_code)]

enum Either {
    One,
    Other(String,String)
}

static one : Either = Either::One;

pub fn main () { }
