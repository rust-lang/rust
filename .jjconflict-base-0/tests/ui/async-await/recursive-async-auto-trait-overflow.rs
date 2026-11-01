// Regression test for <https://github.com/rust-lang/rust/issues/145151>.

//@ edition: 2024
//@ check-pass

async fn process<'a>() {
    Box::pin(process()).await;
}

fn require_send(_: impl Send) {}

fn main() {
    require_send(process());
}
