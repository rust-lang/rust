#![feature(const_trait_impl)]
#![allow(incomplete_features)]

struct Foo;

const impl Foo { //~ ERROR: expected identifier, found keyword
    fn bar() {}
}

fn main() {
     // shouldn't error here because we shouldn't have been able to recover above
     Foo::bar();
}
