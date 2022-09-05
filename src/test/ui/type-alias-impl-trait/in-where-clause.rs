// check-pass
// This previously caused a query cycle as we evaluated
// `1 + 2` with `Reveal::All` during typeck, causing us to
// to get the concrete type of `Bar` while computing it.
#![feature(type_alias_impl_trait)]
type Bar = impl Sized;

fn foo() -> Bar
where
    Bar: Send,
{
    [0; 1 + 2]
}

fn main() {}
