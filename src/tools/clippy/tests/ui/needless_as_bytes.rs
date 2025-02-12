#![warn(clippy::needless_as_bytes)]
#![allow(clippy::const_is_empty)]
#![feature(exact_size_is_empty)]

struct S;

impl S {
    fn as_bytes(&self) -> &[u8] {
        &[]
    }
    fn bytes(&self) -> &[u8] {
        &[]
    }
}

fn main() {
    if "some string".as_bytes().is_empty() {
        //~^ needless_as_bytes
        println!("len = {}", "some string".as_bytes().len());
        //~^ needless_as_bytes
    }
    if "some string".bytes().is_empty() {
        //~^ needless_as_bytes
        println!("len = {}", "some string".bytes().len());
        //~^ needless_as_bytes
    }

    let s = String::from("yet another string");
    if s.as_bytes().is_empty() {
        //~^ needless_as_bytes
        println!("len = {}", s.as_bytes().len());
        //~^ needless_as_bytes
    }
    if s.bytes().is_empty() {
        //~^ needless_as_bytes
        println!("len = {}", s.bytes().len());
        //~^ needless_as_bytes
    }

    // Do not lint
    let _ = S.as_bytes().is_empty();
    let _ = S.as_bytes().len();
    let _ = (&String::new() as &dyn AsBytes).as_bytes().len();
    macro_rules! m {
        (1) => {
            ""
        };
        (2) => {
            "".as_bytes()
        };
    }
    m!(1).as_bytes().len();
    let _ = S.bytes().is_empty();
    let _ = S.bytes().len();
    let _ = (&String::new() as &dyn Bytes).bytes().len();
    macro_rules! m {
        (1) => {
            ""
        };
        (2) => {
            "".bytes()
        };
    }
    m!(1).bytes().len();
    m!(2).len();
}

pub trait AsBytes {
    fn as_bytes(&self) -> &[u8];
}

impl AsBytes for String {
    fn as_bytes(&self) -> &[u8] {
        &[]
    }
}

pub trait Bytes {
    fn bytes(&self) -> &[u8];
}

impl Bytes for String {
    fn bytes(&self) -> &[u8] {
        &[]
    }
}
