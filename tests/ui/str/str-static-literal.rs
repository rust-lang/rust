//! Check that a bare string literal is typed as a `&'static str` and is usable.

//@ run-pass

pub fn main() {
    let x: &'static str = "foo";
    println!("{}", x);
}
