// edition:2018

mod m { pub fn f() {} }
mod n { pub fn g() {} }

pub use m::f;
pub use n::g;
