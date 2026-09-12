//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ edition: 2024

// Regression test for the second example in
// https://github.com/rust-lang/trait-system-refactor-initiative/issues/283.
//
// `rustc_dump_layout` normalizes `Foo` to the coroutine returned by `get`. Coroutine layouts can
// only be computed in `TypingMode::Codegen`, using `TypingMode::PostAnalysis` makes `layout_of`
// return error instead of reporting the expected layout. This was in both trait solvers.

#![feature(rustc_attrs)]
#![feature(type_alias_impl_trait)]
#![expect(internal_features)]

async fn get() {}

#[rustc_dump_layout(size)]
type Foo = impl Sized;
//~^ ERROR: size: Size(1 bytes)

#[define_opaque(Foo)]
fn main() {
    let _: Foo = get();
}
