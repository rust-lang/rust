//@ edition:2021
//@ aux-crate:to_reuse_functions=to-reuse-functions.rs
//@ pretty-mode:hir
//@ pretty-compare-only
//@ pp-exact:delegation-inherit-attributes.pp

#![allow(incomplete_features)]
#![feature(fn_delegation)]

extern crate to_reuse_functions;

mod to_reuse {
    #[must_use = "foo: some reason"]
    #[cold]
    pub fn foo(x: usize) -> usize {
        x
    }

    #[must_use]
    #[cold]
    pub fn foo_no_reason(x: usize) -> usize {
        x
    }

    #[cold]
    #[deprecated]
    pub fn bar(x: usize) -> usize {
        x
    }
}

#[deprecated]
reuse to_reuse::foo as foo1 {
    self + 1
}

reuse to_reuse::foo_no_reason {
    self + 1
}

#[deprecated]
#[must_use = "some reason"]
reuse to_reuse::foo as foo2 {
    self + 1
}

reuse to_reuse::bar;

reuse to_reuse_functions::unsafe_fn_extern;
reuse to_reuse_functions::extern_fn_extern;
reuse to_reuse_functions::const_fn_extern;
#[must_use = "some reason"]
reuse to_reuse_functions::async_fn_extern;

mod recursive {
    // Check that `baz` inherit attribute from `foo`
    mod first {
        fn bar() {}
        #[must_use = "some reason"]
        reuse bar as foo;
        reuse foo as baz;
    }

    // Check that `baz` inherit attribute from `bar`
    mod second {
        #[must_use = "some reason"]
        fn bar() {}

        reuse bar as foo;
        reuse foo as baz;
    }

    // Check that `foo5` don't inherit attribute from `bar`
    // and inherit attribute from foo4, check that foo1, foo2 and foo3
    // inherit attribute from bar
    mod third {
        #[must_use = "some reason"]
        fn bar() {}
        reuse bar as foo1;
        reuse foo1 as foo2;
        reuse foo2 as foo3;
        #[must_use = "foo4"]
        reuse foo3 as foo4;
        reuse foo4 as foo5;
    }

    mod fourth {
        trait T {
            fn foo(&self, x: usize) -> usize { x + 1 }
        }

        struct X;
        impl T for X {}

        #[must_use = "some reason"]
        reuse <X as T>::foo { self + 1 }
        reuse foo as bar { self + 1 }
    }
}

fn main() {}
