// Test that we do not report false positives of a crate as being found multiple times in the
// dependency graph with different versions, when this crate is only present once. In this test,
// this was happening for `crate1` when two different crates in the dependencies were imported
// as ExternCrateSource::Path.
// Issue #148892.
//@ aux-crate:crate1=crate1.rs

struct MyStruct; //~ HELP  the trait `Trait` is not implemented for `MyStruct`

fn main() {
    crate1::foo(MyStruct); //~ ERROR the trait bound `MyStruct: Trait` is not satisfied
}
