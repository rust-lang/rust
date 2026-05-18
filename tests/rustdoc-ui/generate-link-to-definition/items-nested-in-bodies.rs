// When trying to resolve a type-relative path (e.g., `T::Item` where `T` is a type param) in an
// item that's nested inside of a body (e.g., of a function or constant), we once tried to look up
// the definition in the `TypeckResults` of the body owner which is wrong and led to compiler ICEs.
//
// We now make sure to invalidate the `TypeckResults` when crossing a body - item border.
//
// For additional context, `TypeckResults` as returned by queries like `typeck` store the typeck
// results of *bodies* only. In item signatures / non-bodies, there's no equivalent at the time of
// writing, so it's impossible to resolve HIR TypeRelative paths (identified by a `HirId`) to their
// definition (`DefId`) in other parts of the compiler / in tools.

//@ compile-flags: -Zunstable-options --generate-link-to-definition
//@ check-pass
// issue: <https://github.com/rust-lang/rust/issues/147882>

fn scope() {
    struct X<T: Iterator>(T::Item);

    trait Trait {
        type Ty;

        fn func<T: Iterator>()
        where
            T::Item: Copy
        {}
    }

    impl<T: Iterator> Trait for T {
        type Ty = T::Item;
    }
}
