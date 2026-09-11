//@ revisions: current next
//@ [next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/134312>.
// Used to hang

trait Trait {
    type Type<'a>;
}

trait Trait2 {
    type Type2;
}

trait Test {
    type Assoc: Trait
    where
        for<'a> <Self::Assoc as Trait>::Type<'a>: Trait2<Type2 = ()>;
}

fn main() {}
