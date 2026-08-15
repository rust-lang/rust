//@ revisions: without_aob with_aob
//@ edition: 2024
//@ [without_aob] compile-flags: -Znext-solver -Zdxf
//@ [without_aob] known-bug: #126551
//@ [with_aob] compile-flags: -Znext-solver -Zdxf -Zassumptions-on-binders
//@ [with_aob] check-pass

// Minimized from futures `join_all` + `then`/`map` combinators.
//
// `async { &() }` produces a coroutine whose type contains `&'static ()`. After MIR
// region erasure, `'static` becomes `ReErased`, then `coroutine_hidden_types` rebinds it
// as a universally-quantified `BoundVar`. When the auto-trait solver checks `Send` for
// the outer coroutine's witness, it opens the binder and gets `&'!1_0 ()` — a placeholder.
// In erased mode, it can't prove the inner coroutine is `Send`.
//
// With `-Zassumptions-on-binders`, the solver can use the NLL-derived assumption to prove
// `(): '!1_0` because `'!1_0` outlives `'static`, and so is the coroutine `Send`.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

enum MaybeDone<F: Future> {
    Future(F),
    Done(F::Output),
}

struct Map<Fut, F>(Fut, F);

impl<Fut: Future, F: FnOnce(Fut::Output) -> T, T> Future for Map<Fut, F> {
    type Output = T;
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<T> { todo!() }
}

async fn foo() {
    let _md = MaybeDone::Future(Map(async { &() }, |_| async {}));
    async {}.await;
}

fn assert_send<T: Send>(_: T) {}

fn main() {
    assert_send(foo());
}
