#[cfg(before)]
extern crate a;
#[cfg(after)]
extern crate a;
extern crate b;
extern crate c;

fn t(a: &'static usize) -> usize {
    a as *const _ as usize
}

fn main() {
    assert_eq!(t(a::token()), t(b::a_token()));
    assert!(t(a::token()) != t(c::a_token()));
}
