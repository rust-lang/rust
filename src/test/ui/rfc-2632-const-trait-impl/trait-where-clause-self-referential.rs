// check-pass

#![feature(const_trait_impl)]
#![feature(const_fn_trait_bound)]

trait Foo {
    fn bar() where Self: ~const Foo;
}

struct S;

impl Foo for S {
    fn bar() {}
}

fn baz<T: Foo>() {
    T::bar();
}

const fn qux<T: ~const Foo>() {
    T::bar();
}

fn main() {}
