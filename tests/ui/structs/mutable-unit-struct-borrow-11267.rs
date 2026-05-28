//! Regression test for https://github.com/rust-lang/rust/issues/11267

//@ run-pass
// Tests that unary structs can be mutably borrowed.

struct Empty;

trait T<U> {
    fn next(&mut self) -> Option<U>;
}
impl T<isize> for Empty {
    fn next(&mut self) -> Option<isize> { None }
}

fn do_something_with(a : &mut dyn T<isize>) {
    println!("{:?}", a.next())
}

pub fn main() {
    do_something_with(&mut Empty);
}
