#![feature(const_generics)]
#![allow(incomplete_features)]
#![deny(unused_lifetimes)]

fn err<'a>() -> [u8; 3 + 4] { todo!() }
//~^ ERROR lifetime parameter `'a` never used

fn hrtb_err() where for<'a> [u8; 3 + 4]: Sized {}
// FIXME(const_generics): This should error

fn ok1<'a>() -> [u8; { let _: &'a (); 3 }] { todo!() }

fn ok2<'a>() -> [u8; 3 + 4] {
    let _: &'a ();
    todo!()
}

fn hrtb_ok() where for<'a> [u8; { let _: &'a (); 3 }]: Sized {}

fn main() {}
