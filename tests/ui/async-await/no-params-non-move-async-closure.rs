//@ edition:2018
//@ check-pass

#![feature(async_closure)]

fn main() {
    let _ = async |x: u8| {};
}
