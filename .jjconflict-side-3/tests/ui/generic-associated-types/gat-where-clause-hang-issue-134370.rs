//@ revisions: current next
//@ [next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/134370>.
// Used to get stuck in trait solver

trait Assoc {
    type Ty;
}

trait Foo {
    type Gat<'a>;
}

trait Bar {
    type Ty: Foo
    where
        for<'a> <Self::Ty as Foo>::Gat<'a>: Assoc<Ty = ()>;
}

fn main() {}
