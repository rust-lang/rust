#[cfg(before)] extern crate a;
extern crate b;
extern crate c;
#[cfg(after)] extern crate a;

fn t(a: &'static uint) -> uint { a as *const _ as uint }

fn main() {
    assert!(t(a::token()) == t(b::a_token()));
    assert!(t(a::token()) != t(c::a_token()));
}
