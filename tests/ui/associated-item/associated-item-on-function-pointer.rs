// Check that we successfully resolve associated functions and constants defined on
// function pointer types. Regression test for issue #108270.

// check-pass

#![feature(rustc_attrs)]
#![rustc_coherence_is_core]

impl fn() {
    const ARITY: usize = 0;

    fn handle() {}

    fn apply(self) {
        self()
    }
}

impl for<'src> fn(&'src str) -> bool {
    fn execute(self, source: &'static str) -> bool {
        self(source)
    }
}

fn main() {
    let _: usize = <fn()>::ARITY;
    <fn()>::handle();

    let f: fn() = main;
    f.apply();

    let predicate: fn(&str) -> bool = |source| !source.is_empty();
    let _ = predicate.execute("...");
}
