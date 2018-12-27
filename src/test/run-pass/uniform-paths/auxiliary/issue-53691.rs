// edition:2018

#![feature(uniform_paths)]

mod m { pub fn f() {} }
mod n { pub fn g() {} }

pub use m::f;
pub use n::g;
