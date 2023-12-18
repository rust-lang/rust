// known-bug: #110395
// failure-status: 101
// dont-check-compiler-stderr

// const closures don't have host params...

#![feature(const_closures, const_trait_impl, effects)]
#![allow(incomplete_features)]

trait Foo {
    fn foo(&self);
}

impl Foo for () {
    fn foo(&self) {}
}

fn main() {
    (const || { (()).foo() })();
}
