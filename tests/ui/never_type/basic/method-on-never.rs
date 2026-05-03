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
    //~^ WARN [trait_method_on_coerced_never_type]
    //~| WARN previously accepted

    { loop {} }.method();
    //~^ WARN [trait_method_on_coerced_never_type]
    //~| WARN previously accepted
}
