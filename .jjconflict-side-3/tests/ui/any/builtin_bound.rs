//@ revisions: next old
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass
#![feature(try_as_dyn)]

trait Trait {}

// In contrast to `T: Sized`, `Struct<T>: Sized` does not go
// through a fast path, and is thus rejected by the builtin impl
// check that rejects all builtin impls in reflection mode.
// FIXME(try_as_dyn): should probably allow builtin impls that
// are never lifetime dependent (like Sized).
impl<T> Trait for Struct<T> where Struct<T>: Sized {}

struct Struct<T>(T);

const _: () = {
    let x = Struct(42);
    assert!(std::any::try_as_dyn::<_, dyn Trait>(&x).is_none());
};

fn main() {}
