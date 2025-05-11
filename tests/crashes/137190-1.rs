//@ known-bug: #137190
//@ compile-flags: -Zmir-opt-level=2 -Zvalidate-mir
trait A {
    fn b(&self);
}
trait C: A {}
impl C for () {}
fn main() {
    (&() as &dyn C as &dyn A).b();
}
