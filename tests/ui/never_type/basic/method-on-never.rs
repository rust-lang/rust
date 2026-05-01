//@ check-pass
// Regression test for https://github.com/rust-lang/rust/issues/143349

#![feature(never_type)]

trait Trait {
    fn method(&self);
}
impl Trait for ! {
    fn method(&self) {
        todo!()
    }
}

fn main() {
    let x = loop {};
    x.method();

    { loop {} }.method();
}
