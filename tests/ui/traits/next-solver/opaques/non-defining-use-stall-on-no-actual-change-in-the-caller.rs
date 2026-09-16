//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// This is rather an implementation-detail-related test.
//
// We track self-bounds of opaque-hidden-types in the `InferCtxt` and pass them via both query
// inputs and responses. We try to dedup them per eager resolving and structural equality but they
// don't always work well, especially due to canonicalizations.
//
// Suppose that we have such bound with `'static` lifetime. If we try to normalize that hidden type,
// we get that bound from the query response and is registered into the caller's context. And if we
// try to reevaluate that nested normalization goal again, it is passed into a query input. But as
// `'static` region is canonicalized into a placeholder region, we cannot deduplicate it in the
// callee's side and thus we return the same bound with `'static` region from the previous region
// again.
//
// This pathetic reevaluation never stops and until we hit the recursion limit and end up with an
// overflow. To prevent this, we simply check whether the number of opaque hidden ty bounds has
// actually increased from the evaluation after instantiating the response from the callers side to
// decide evaluation's `has_changed`.

#![allow(warnings)]

fn features() -> impl Iterator<Item = &'static ()> {
    None.into_iter()
}

trait Foo {
    fn foo(&self) {}
}

trait Qux {
    type Assoc: Foo;

    fn qux(&self) -> Self::Assoc {
        loop {}
    }
}

trait Bar {
    type Assoc: Qux<Assoc = &'static ()>;

    fn bar(&self) -> Self::Assoc {
        loop {}
    }
}

trait Baz {
    type Assoc: Bar;

    fn baz(&self) -> Self::Assoc {
        loop {}
    }
}

impl Foo for &'static () {}

impl Qux for () {
    type Assoc = &'static ();
}

impl Bar for () {
    type Assoc = ();
}

impl Baz for () {
    type Assoc = ();
}

fn heck() -> impl Baz {
    heck().baz().bar().qux().foo()
}

fn main() {}
