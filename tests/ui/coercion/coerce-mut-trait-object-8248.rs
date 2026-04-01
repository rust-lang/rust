// https://github.com/rust-lang/rust/issues/8248
//@ run-pass

trait A {
    fn dummy(&self) { } //~ WARN method `dummy` is never used
}
struct B;
impl A for B {}

fn foo(_: &mut dyn A) {}

pub fn main() {
    let mut b = B;
    foo(&mut b as &mut dyn A);
}
