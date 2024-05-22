// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// skip-filecheck
//@ test-mir-pass: Inline
//@ edition: 2021
//@ compile-flags: -Zinline-mir-hint-threshold=10000 -Zinline-mir-threshold=10000 --crate-type=lib

pub async fn run(permit: ActionPermit<'_, ()>, ctx: &mut core::task::Context<'_>) {
    run2(permit, ctx);
}

// EMIT_MIR inline_coroutine_body.run2-{closure#0}.Inline.diff
fn run2<T>(permit: ActionPermit<'_, T>, ctx: &mut core::task::Context) {
    _ = || {
        let mut fut = ActionPermit::perform(permit);
        let fut = unsafe { core::pin::Pin::new_unchecked(&mut fut) };
        _ = core::future::Future::poll(fut, ctx);
    };
}

pub struct ActionPermit<'a, T> {
    _guard: core::cell::Ref<'a, T>,
}

impl<'a, T> ActionPermit<'a, T> {
    async fn perform(self) {
        core::future::ready(()).await
    }
}
