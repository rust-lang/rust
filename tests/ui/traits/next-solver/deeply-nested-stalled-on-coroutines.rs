//@ compile-flags: -Znext-solver
//@ check-pass
//@ edition: 2024

// Regression test for https://github.com/rust-lang/trait-system-refactor-initiative/issues/270

struct Wrap<T>(T);

impl<T> Wrap<T> {
    fn nest(self) -> Wrap<Self> {
        Wrap(self)
    }
}

fn assert_send<S: Send>(s: S) -> S {
    s
}

// We need an indirection so that the goal stalled on coroutine is not the root goal
fn mk_opaque<F>(f: F) -> impl Future
where
    F: AsyncFnOnce(),
{
    f()
}

fn test() {
    let coroutine = async || {};
    let opaque = mk_opaque(coroutine);
    // This `deep_nested: Send` is ambiguous because it contains nested obligation whose self_ty is
    // coroutine.
    // It should be collected into `stalled_coroutine_obligation` before report ambiguity errors.
    // But it used to be not, because we were collecting them with a `ProofTreeVisitor`, which has
    // a recursion limit.
    let deep_nested =
        Wrap(opaque).nest().nest().nest().nest().nest().nest().nest().nest().nest().nest().nest();

    assert_send(deep_nested);
}

fn main() {}
