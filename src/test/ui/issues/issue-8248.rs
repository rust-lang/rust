// run-pass
// pretty-expanded FIXME #23616

trait A {
    fn dummy(&self) { }
}
struct B;
impl A for B {}

fn foo(_: &mut dyn A) {}

pub fn main() {
    let mut b = B;
    foo(&mut b as &mut dyn A);
}
