#![feature(non_ascii_idents)]

#[no_mangle]
pub fn řųśť() {}  //~ `#[no_mangle]` requires ASCII identifier

fn main() {}
