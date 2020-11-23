// check-pass
#![feature(const_generics)]
#![allow(incomplete_features)]
#![deny(unused_lifetimes)]

// FIXME(const_generics): this should error
fn test<'a>() -> [u8; 3 + 4] { todo!() }

fn ok1<'a>() -> [u8; { let _: &'a (); 3 }] { todo!() }

fn ok2<'a>() -> [u8; 3 + 4] {
    let _: &'a ();
    todo!()
}

fn main() {}
