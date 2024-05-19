// Check that the hash for `mod3::bar` changes when we change the
// `use` to something different.

//@ revisions: rpass1 rpass2 rpass3
//@ compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]

fn test<T>() { }

mod mod1 {
    pub struct Foo(pub u32);
}

mod mod2 {
    pub struct Foo(pub i64);
}

mod mod3 {
    #[cfg(rpass1)]
    use mod1::Foo;
    use test;

    // In rpass2 we move the use declaration.
    #[cfg(rpass2)]
    use mod1::Foo;

    // In rpass3 we let the declaration point to something else.
    #[cfg(rpass3)]
    use mod2::Foo;

    #[rustc_clean(cfg="rpass2")]
    #[rustc_clean(except="opt_hir_owner_nodes,typeck", cfg="rpass3")]
    fn in_expr() {
        Foo(0);
    }

    #[rustc_clean(cfg="rpass2")]
    #[rustc_clean(except="opt_hir_owner_nodes,typeck", cfg="rpass3")]
    fn in_type() {
        test::<Foo>();
    }
}

fn main() { }
