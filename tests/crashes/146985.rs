//@ known-bug: rust-lang/rust#146985
//@edition: 2024

fn invalid_future() -> impl Future {
    create_complex_future()
}

fn create_complex_future() -> impl Future<Output = impl ReturnsSend> {
    async { &|| async { invalid_future().await } }
}

fn coerce_impl_trait() -> impl Future<Output = impl Send> {
    create_complex_future()
}

trait ReturnsSend {}

impl<F, R> ReturnsSend for F
where
    F: Fn() -> R,
    R: Send,
{
}

fn main() {}
