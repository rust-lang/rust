//! We evaluate `1 + 2` with `Reveal::All` during typeck, causing
//! us to get the concrete type of `Bar` while computing it.
//! This again requires type checking `foo`.
#![feature(type_alias_impl_trait)]
type Bar = impl Sized;
//~^ ERROR: cycle

fn foo() -> Bar
where
    Bar: Send,
{
    [0; 1 + 2]
    //~^ ERROR: type annotations needed: cannot satisfy `Bar: Send`
}

fn main() {}
