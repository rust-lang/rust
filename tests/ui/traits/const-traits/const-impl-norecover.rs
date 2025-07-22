#![feature(const_trait_impl)]

struct Foo;

const impl Foo { //~ ERROR: inherent impls cannot be const
    fn bar() {}
}

fn main() {
     // shouldn't error here because we shouldn't have been able to recover above
     Foo::bar();
}
