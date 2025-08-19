//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ edition: 2024
//@[current] check-pass

// This previously caused an ICE with the new solver.
// The delayed coroutine obligations were checked with the
// opaque types inferred by borrowck.
//
// One of these delayed obligations failed with overflow in
// borrowck, causing us to taint `type_of` for the opaque. This
// then caused us to also not emit an error when checking the
// coroutine obligations.

fn build_multiple<'a>() -> impl Sized {
    spawn(async { build_dependencies().await });
    //[next]~^ ERROR overflow evaluating the requirement
}

// Adding an explicit `Send` bound fixes it.
// Proving `build_dependencies(): Send` in `build_multiple` adds
// addiitional defining uses/placeholders.
fn build_dependencies() -> impl Future<Output = ()> /* + Send */ {
    async {
        Box::pin(build_dependencies()).await;
        async { build_multiple() }.await;
    }
}

fn spawn<F: Send>(_: F) {}

fn main() {}
