//@ revisions: current next
//@[current] check-pass
//@[next] compile-flags: -Znext-solver
//@[next] check-fail
//@ ignore-compare-mode-next-solver (explicit revisions)

/// This triggers an ICE with (and without) `--emit metadata` using the old
/// trait solver:
/// ```
/// rustc +nightly-2023-01-09 \
///   tests/ui/higher-ranked/trait-bounds/rigid-equate-projections-in-higher-ranked-fn-signature.rs
/// ```
/// The ICE was unknowingly fixed by
/// <https://github.com/rust-lang/rust/pull/101947> in `nightly-2023-01-10`.
/// This is a regression test for that fixed ICE. For the next solver we simply
/// make sure there is a compiler error.

trait Trait<'a> {
    type Assoc;
}

fn foo<T: for<'a> Trait<'a>>() -> for<'a> fn(<T as Trait<'a>>::Assoc) {
    todo!()
}

fn bar<T: for<'a> Trait<'a>>() {
    let _: for<'a> fn(<_ as Trait<'a>>::Assoc) = foo::<T>(); //[next]~ ERROR type annotations needed
}

fn main() {}
