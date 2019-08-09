// Check that the hash for a method call is sensitive to the traits in
// scope.

// revisions: rpass1 rpass2
// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]

fn test<T>() { }

trait Trait1 {
    fn method(&self) { }
}

impl Trait1 for () { }

trait Trait2 {
    fn method(&self) { }
}

impl Trait2 for () { }

mod mod3 {
    #[cfg(rpass1)]
    use Trait1;
    #[cfg(rpass2)]
    use Trait2;

    #[rustc_clean(label="Hir", cfg="rpass2")]
    #[rustc_clean(label="HirBody", cfg="rpass2")]
    #[rustc_dirty(label="typeck_tables_of", cfg="rpass2")]
    fn bar() {
        ().method();
    }

    #[rustc_clean(label="Hir", cfg="rpass2")]
    #[rustc_clean(label="HirBody", cfg="rpass2")]
    #[rustc_clean(label="typeck_tables_of", cfg="rpass2")]
    fn baz() {
        22; // no method call, traits in scope don't matter
    }
}

fn main() { }
