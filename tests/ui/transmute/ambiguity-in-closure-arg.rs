//! Minimized test for <https://github.com/rust-lang/rust/issues/123461>.

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [next] compile-flags: -Znext-solver
//@ check-fail

struct Unconstrained<T>(T);

fn main() {
    unsafe { std::mem::transmute::<_, ()>(|o_b: Unconstrained<_>| {}) };
    //~^ ERROR type annotations needed
    // We unfortunately don't check `Wf(Unconstrained<_>)`, so we won't
    // hit an ambiguity error before checking the transmute. That means
    // we still may have inference variables in our transmute src.
}
