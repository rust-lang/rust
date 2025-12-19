#![feature(extern_item_impls)]
// EIIs can, despite not being super useful, be declared in statement position
// nested inside items. Items in statement position, when expanded as part of a macro,
// need to be wrapped slightly differently (in an `ast::Statement`).
// We did this on the happy path (no errors), but when there was an error, we'd
// replace it with *just* an `ast::Item` not wrapped in an `ast::Statement`.
// This caused an ICE (https://github.com/rust-lang/rust/issues/149980).
// this test fails to build, but demonstrates that no ICE is produced.

fn main() {
    struct Bar;

    #[eii]
    //~^ ERROR `#[eii]` is only valid on functions
    impl Bar {}
}
