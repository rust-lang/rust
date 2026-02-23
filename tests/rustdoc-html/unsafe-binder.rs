//@ aux-build:unsafe-binder-dep.rs

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

extern crate unsafe_binder_dep;

//@ has 'unsafe_binder/fn.woof.html' //pre "fn woof() -> unsafe<'a> &'a str"
pub use unsafe_binder_dep::woof;

//@ has 'unsafe_binder/fn.meow.html' //pre "fn meow() -> unsafe<'a> &'a str"
pub fn meow() -> unsafe<'a> &'a str { todo!() }

//@ has 'unsafe_binder/fn.meow_squared.html' //pre "fn meow_squared() -> unsafe<'b, 'a> &'a &'b str"
pub fn meow_squared() -> unsafe<'b, 'a> &'a &'b str { todo!() }
