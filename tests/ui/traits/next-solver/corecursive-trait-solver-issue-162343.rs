//@ build-fail
//@ compile-flags: --crate-type=lib -Znext-solver=globally -Awarnings
// Regression test for #162343.
// Resolving baz during codegen hits the recursive RawMessage bound.
// This used to ICE instead of reporting the overflow at the impl.

trait RawMessage<'a> {
    fn baz() {}
}

impl<'a> RawMessage<'a> for () //~ ERROR overflow evaluating the requirement
where
    (): RawMessage<'a>,
{}

trait EmptyState2 {
    fn bar() {}
}

impl<'a> EmptyState2 for ()
where
    (): RawMessage<'a>,
{
    fn bar() {
        <() as RawMessage>::baz();
    }
}
