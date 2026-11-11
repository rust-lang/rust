//@ compile-flags: -Znext-solver
//@ edition: 2024
//@ check-pass

// Regression test for trait-system-refactor-initiative#282
// Previously we always computed higher ranked assumptions for coroutines.
// It led to query cycle in recursive functions like the one below.
// Thoses assumptions are not used in the next solver and
// the functionality is superseded by `assumptions-on-binders`.

fn go() -> impl Future + Send + 'static {
    spawn(async {
        go().await;
    })
}
fn spawn(_: impl Future + Send + 'static) -> impl Future {
    async {}
}

fn main() {}
