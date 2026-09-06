//@ edition: 2024
//@ revisions: current next
//@ [next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/160355>.
// Used to ICE

struct Wrap<F>(F);

impl<F> Service for Wrap<F> where F: AsyncFn() {}

trait Service {}

impl<F> Service for F where F: AsyncFn() {}

async fn ice<P>() {}

fn main() {
    let service = Wrap(ice::<&'static ()>);
    let _ = Box::new(service) as Box<dyn Service>;
}
