//@ revisions: current next
//@[current] check-pass
//@[next] compile-flags: -Znext-solver
//@[next] check-fail
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ edition: 2021

/// Regression test for <https://github.com/rust-lang/rust/issues/162559>:
/// compiling `wasmtime-wasi-http@48.0.1` fails with `E0283` under the next
/// trait solver.
///
/// Minimized from the `wasmtime::component::bindgen!`-generated code for the
/// `wasi:http/service` world: `HasData` stands in for
/// `wasmtime::component::HasData`, the `fn(&mut _) -> D2::Data<'_>` argument
/// is the host getter passed to `Accessor::with_getter`, and the
/// `for<'a> D2::Data<'a>: Tr` obligation is one of the generated
/// `HostRequestWithStore<T>`-shaped bounds.
///
/// The old solver eagerly equates `<D2 as HasData>::Data<'a>` with
/// `<D as HasData>::Data<'a>` and infers `D2 = D`, while the next solver
/// deliberately does not relate unresolved higher-ranked aliases (see
/// <https://github.com/rust-lang/trait-system-refactor-initiative/issues/168>),
/// so this is rejected with `E0283`. Wasmtime fixed the generated code to pass
/// explicit generic arguments in
/// <https://github.com/bytecodealliance/wasmtime/pull/14193>; for the next
/// solver we simply make sure there is a compiler error with an actionable
/// suggestion.

trait HasData {
    type Data<'a>;
}

trait Tr {}

fn use_host<D2: HasData>(f: fn(&mut ()) -> D2::Data<'_>)
where
    for<'a> D2::Data<'a>: Tr,
{
}

fn generated<D: HasData>(getter: fn(&mut ()) -> D::Data<'_>)
where
    for<'a> D::Data<'a>: Tr,
{
    use_host(getter); //[next]~ ERROR type annotations needed
}

fn main() {}
