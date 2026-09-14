//@ compile-flags: -Znext-solver
//@ build-pass
//@ edition: 2021
//@ compile-flags: -C debuginfo=1 --crate-type=lib
// This test explores a funny case when handling opaque types:
// We involve a couple of iterator types from the standard library, which
// importantly is another crate from this test's perspective.
// When we check whether we can normalize something, if the DefId is from
// another crate we refuse to.
//
// However, in `TypingMode::PostAnalysis`, all opaques become normalizable,
// and so also these iteraators from `std`. In the next solver, when we
// canonicalize, we always do a first attempt in `TypingMode::ErasedNotCoherence`,
// deleting all opaque types in scope.
//
// If we then end up accessing opaques, we rerun the canonicalization in the
// original typing modes *with* opaques. This relies on us properly tracking
// whether opaques were used. And, even if opaque types have a non-local defid,
// we *can* end up being able to normalize said opaque if the original `TypingMode`
// was `PostAnalysis`.
//
// This test makes sure that we indeed track these opaque type accesses properly.

pub(crate) struct Foo;

impl From<()> for Foo {
    fn from(_: ()) -> Foo {
        String::new().extend('a'.to_uppercase());
        Foo
    }
}
