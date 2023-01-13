// run-pass
#![allow(unused_must_use)]
use std::slice::Iter;
use std::io::{Error, ErrorKind, Result};
use std::vec::*;

fn foo(it: &mut Iter<u8>) -> Result<u8> {
    Ok(*it.next().unwrap())
}

fn bar() -> Result<u8> {
    let data: Vec<u8> = Vec::new();

    if true {
        return Err(Error::new(ErrorKind::NotFound, "msg"));
    }

    let mut it = data.iter();
    foo(&mut it)
}

fn main() {
    bar();
}
