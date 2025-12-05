#![feature(const_closures, const_trait_impl)]
#![allow(incomplete_features)]

trait Foo {
    fn foo(&self);
}

impl Foo for () {
    fn foo(&self) {}
}

fn main() {
    (const || { (()).foo() })();
    //~^ ERROR: }: [const] Fn()` is not satisfied
}
