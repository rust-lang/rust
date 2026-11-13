// Regression test for <https://github.com/rust-lang/rust/issues/151299>.
// An invalid `impl Future` awaited inside a nested async block used to ICE with
// "Failed to normalize" in normalize_erasing_regions.
//@ edition: 2024
//@ compile-flags: -Zvalidate-mir -Znext-solver=globally

fn invalid_future() -> impl Future {}
//~^ ERROR `()` is not a future

fn create_complex_future() -> impl Future {
    async { &|| async { invalid_future().await } }
}

fn main() {}
