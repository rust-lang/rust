//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #147719
//@ edition: 2024

// A variant of `old-solver-only-leak-check-on-ambig.rs` which
// relies on us deferring the inference of async closure upvars.
// `Fut` is higher ranked as it ends up capturing the function
// argument.

struct Wrap<F>(F);
trait NotImplemented {}
trait NodeImpl {}
impl<T: NotImplemented> NodeImpl for T {}
impl<F, Fut> NodeImpl for Wrap<F>
where
    F: Fn(&()) -> Fut,
{
}

fn node_impl<T: NodeImpl>(_: T) {}
fn main() {
    node_impl(Wrap(async |_: &()| ()));
    //[next]~^ ERROR: the trait bound
}
