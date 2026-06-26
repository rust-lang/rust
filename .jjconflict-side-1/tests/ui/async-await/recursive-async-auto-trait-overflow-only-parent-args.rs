// Regression test for #145288. This is the same issue as #145151
// which we fixed in #145194. However in that PR we accidentally created
// a `CoroutineWitness` which referenced all generic arguments of the
// coroutine, including upvars and the signature.

//@ edition: 2024
//@ check-pass

async fn process<'a>(x: &'a u32) {
    Box::pin(process(x)).await;
}

fn require_send(_: impl Send) {}

fn main() {
    require_send(process(&1));
}
