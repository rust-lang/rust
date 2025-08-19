// Repro for <https://github.com/rust-lang/rust/issues/126350#issue-2349492101>.
//@ edition: 2021
//@ revisions: assumptions no_assumptions
//@[assumptions] compile-flags: -Zhigher-ranked-assumptions
//@[assumptions] known-bug: unknown
//@[no_assumptions] known-bug: #110338

fn assert_send<T: Send>(_: T) {}

#[derive(Clone)]
struct Ctxt<'a>(&'a ());

async fn commit_if_ok<'a>(ctxt: &mut Ctxt<'a>, f: impl AsyncFnOnce(&mut Ctxt<'a>)) {
    f(&mut ctxt.clone()).await;
}

fn operation(mut ctxt: Ctxt<'_>) {
    assert_send(async {
        commit_if_ok(&mut ctxt, async |_| todo!()).await;
    });
}

fn main() {}
