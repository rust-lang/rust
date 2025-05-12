//@ aux-build:first_crate.rs
//@ aux-build:second_crate.rs
//@ revisions:rpass1 rpass2

// Regression test for issue #92163
// Under certain circumstances, we may end up trying to
// decode a foreign `Span` from the incremental cache, without previously
// having imported the `SourceFile`s from the owning crate. This can happen
// if the `Span` comes from a transitive dependency (so we never try to resolve
// items from the crate during expansion/resolution).
//
// Previously, this would result in an ICE, since we would not have loaded
// the corresponding `SourceFile` for the `StableSourceFileId` we decoded.
// This test verifies that the decoding of a foreign `Span` will always
// try to import the `SourceFile`s from the foreign crate, instead of
// relying on that having already happened during expansion.

extern crate second_crate;

pub struct Outer;

impl Outer {
    pub fn use_it() {
        // This returns `first_crate::Foo`, causing
        // us to encode the `AdtDef `first_crate::Foo` (along with its `Span`s)
        // into the query cache for the `TypeckResults` for this function.
        second_crate::make_it();
    }
}

fn main() {}
