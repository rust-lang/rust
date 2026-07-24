//! Regression test for #157152.
//!
//! Under `min_generic_const_args` with `macroless_generic_const_args`, a braced const
//! argument containing an associated-function call (e.g. `FieldName::len()`, as generated
//! by `tracing`'s logging macros as `FieldName<{ FieldName::len(name) }>`) was lowered as
//! a tuple-struct constructor. Lowering the callee's `Self` type `FieldName`, written
//! without its `const N: usize` argument, then produced a spurious "missing generics"
//! error (E0107) plus follow-on errors, which made `tracing` fail to compile in any crate
//! enabling the feature.
//!
//! It should instead report that the call must be wrapped in a `const` block, and
//! the wrapped form must compile. The same holds for any self type that cannot host a
//! tuple-variant constructor (unions, primitives, foreign types), not just structs.
//@ compile-flags: -Znext-solver

#![feature(min_generic_const_args, macroless_generic_const_args)]
#![feature(generic_const_args)]
#![feature(extern_types)]
#![expect(incomplete_features)]

struct FieldName<const N: usize>([u8; N]);

impl FieldName<0> {
    const fn len() -> usize {
        5
    }

    const fn len_of(name: &str) -> usize {
        name.len()
    }
}

// The associated-function call is not a constructor, so the bare braces are
// rejected with a clear diagnostic instead of a spurious "missing generics" error.
fn bad(_: FieldName<{ FieldName::len() }>) {}
//~^ ERROR complex const arguments must be placed inside of a `const` block

// Wrapping the call in a `const` block makes it an anonymous const and compiles.
fn good(_: FieldName<{ const { FieldName::len() } }>) {}

// The exact shape from #157152: `tracing`'s macros expand a field name to
// `FieldName::len(stringify!(field))`. Same as `bad` but with a string argument, which
// the diagnostic ignores; the self type is still a bare generic struct.
fn bad_tracing(_: FieldName<{ FieldName::len_of("id") }>) {}
//~^ ERROR complex const arguments must be placed inside of a `const` block

fn good_tracing(_: FieldName<{ const { FieldName::len_of("id") } }>) {}

union Tag<const N: usize> {
    bytes: [u8; N],
}

impl Tag<0> {
    const fn width() -> usize {
        7
    }
}

// Unions behave exactly like structs: the call is an associated function, not a
// constructor, so the bare braces are rejected the same way.
fn bad_union(_: Tag<{ Tag::width() }>) {}
//~^ ERROR complex const arguments must be placed inside of a `const` block

fn good_union(_: Tag<{ const { Tag::width() } }>) {}

// A primitive can't host a constructor either, and has no generics to omit, so it never
// hits the "missing generics" path. No `good_` counterpart: `from_str_radix` returns a
// `Result`, not a `usize`, so the wrapped form can't form a valid const arg. This case
// only checks that the bare form is rejected.
fn bad_prim(_: FieldName<{ u32::from_str_radix("10", 10) }>) {}
//~^ ERROR complex const arguments must be placed inside of a `const` block

unsafe extern "C" {
    type Opaque;
}

// A foreign type has no constructor and no inherent associated functions. The guard
// rejects it from the self type's resolution alone, before the `foo` segment is resolved.
// Without that, downstream resolution gives an opaque "invalid base path" error (plus an
// E0223) rather than this clear one.
fn bad_foreign(_: FieldName<{ Opaque::foo() }>) {}
//~^ ERROR complex const arguments must be placed inside of a `const` block

fn main() {}
