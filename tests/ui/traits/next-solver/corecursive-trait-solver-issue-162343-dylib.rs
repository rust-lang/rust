//@ check-pass
//@ edition: 2021
//@ compile-flags: --crate-type=dylib -Znext-solver=globally -Clink-dead-code
//@ needs-dynamic-linking
//@ needs-crate-type: dylib

#![allow(unused)]

trait RawMessage<'a> {
    fn baz() {}
}

impl<'a> RawMessage<'a> for ()
where
    (): RawMessage<'a>,
{}

pub trait EmptyState2 {
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
