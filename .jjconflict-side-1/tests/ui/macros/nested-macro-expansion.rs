//! Test nested macro expansion with concat! macros

//@ run-pass

static FOO : &'static str = concat!(concat!("hel", "lo"), "world");

pub fn main() {
    assert_eq!(FOO, "helloworld");
}
