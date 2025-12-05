//@ compile-flags: -Znext-solver
//@ check-pass

trait Foo {}

impl<T> Foo for T {}

trait Bar {}

struct Wrapper<'a, T>(&'a T);

impl<'a, T> Bar for Wrapper<'a, T> where &'a T: Foo {}
// We need to satisfy `&'a T: Foo` when checking that this impl is WF
// that can either be satisfied via the param-env, or via an impl.
//
// When satisfied via the param-env, since each lifetime is canonicalized
// separately, we end up getting extra region constraints.
//
// However, when satisfied via the impl, there are no region constraints,
// and we can short-circuit a response with no external constraints.

fn main() {}
