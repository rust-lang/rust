//~ ERROR the type `<() as StaticTy>::Item<'a>` does not fulfill the required lifetime
//@ compile-flags: -Znext-solver
// Regression test for rust-lang/trait-system-refactor-initiative#59

trait StaticTy {
    type Item<'a>: 'static;
}

impl StaticTy for () {
    type Item<'a> = &'a ();
}

fn main() {}
