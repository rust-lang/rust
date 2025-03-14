//@ known-bug: #136381
//@ compile-flags: -Zvalidate-mir -Zmir-enable-passes=+GVN
#![feature(trait_upcasting)]

trait A {}
trait B: A {
    fn c(&self);
}
impl B for i32 {
    fn c(self) {
        todo!();
    }
}

fn main() {
    let baz: &dyn B = &1;
    let bar: &dyn A = baz;
}
