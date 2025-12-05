//! Test that we do not need to handle host effects in `expand_trait_aliases`

#![feature(trait_alias, const_trait_impl)]
//@ check-pass

mod foo {
    pub const trait Bar {
        fn bar(&self) {}
    }
    pub const trait Baz {
        fn baz(&self) {}
    }

    impl const Bar for () {}
    impl const Baz for () {}

    pub const trait Foo = [const] Bar + Baz;
}

use foo::Foo as _;


const _: () = {
    // Ok via `[const] Bar` on `Foo`
    ().bar();
    // Also works, because everything is fully concrete, so we're ignoring that
    // `Baz` is not a const trait bound of `Foo`.
    ().baz();
};

fn main() {}
