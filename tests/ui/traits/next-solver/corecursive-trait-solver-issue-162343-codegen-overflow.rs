//~ ERROR overflow evaluating the requirement `(): RawMessage<'_>`
//@ build-fail
//@ compile-flags: --crate-type=lib -Znext-solver=globally -Clink-dead-code -Awarnings

trait RawMessage<'a> {
    fn baz() {}
}

impl<'a> RawMessage<'a> for ()
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
