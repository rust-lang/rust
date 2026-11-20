//@ edition: 2024
//@ compile-flags: -Znext-solver
//@ check-pass

// Previously we update the rerun flag when we normalize opaques and don't immedidately bail.
// We normalize goals when adding them which is separate from their evaluation.
// So we don't have the `might_rerun` flag when evaluating them.
// This leads to query cycle in code where we need to leak opaque types for auto traits.

fn is_send<T: Send>(_: T) {}

fn inner() -> impl Sized {
    is_send(outer());
}

fn outer() -> impl Sized {
    inner()
}

// From #135062 which was fixed but broken again.
async fn foo() {
    is_send(bar())
}

async fn bar() {
    foo().await;
}

fn main() {}
