//@ edition: 2024
//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

fn create_complex_future() -> impl Future<Output = impl ReturnsSend> {
    //[current]~^ ERROR the trait bound `(): ReturnsSend` is not satisfied
    async { create_complex_future().await }
    //[current]~^ ERROR recursion in an async block requires
    //[next]~^^ ERROR type annotations needed
}

trait ReturnsSend {}
fn main() {}
