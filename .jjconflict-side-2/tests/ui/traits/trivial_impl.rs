//! This test checks that we do need to implement
//! all members, even if their where bounds only hold
//! due to other impls.

trait Foo<T> {
    fn foo()
    where
        Self: Foo<()>;
}

impl Foo<()> for () {
    fn foo() {}
}

impl Foo<u32> for () {}
//~^ ERROR: not all trait items implemented, missing: `foo`

fn main() {}
