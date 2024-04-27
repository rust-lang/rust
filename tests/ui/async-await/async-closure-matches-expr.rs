//@ build-pass
//@ edition:2018

#![feature(async_closure)]

macro_rules! match_expr {
    ($x:expr) => {}
}

fn main() {
    match_expr!(async || {});
}
