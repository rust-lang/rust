//@ check-pass
//@ compile-flags: -Znext-solver
//@ edition: 2024

// A regression test for trait-system-refactor-initiative#245.
// The old solver doesn't check the supertraits of the principal trait
// when considering object candidate for normalization.
// And the new solver previously did, resulting in a placeholder error
// while normalizing inside of a generator witness.

trait AsyncFn: Send + 'static {
    type Fut: Future<Output = ()> + Send;

    fn call(&self) -> Self::Fut;
}

type BoxFuture<'a, T> = std::pin::Pin<Box<dyn Future<Output = T> + Send + 'a>>;
type DynAsyncFnBoxed = dyn AsyncFn<Fut = BoxFuture<'static, ()>>;

fn wrap_call<P: AsyncFn + ?Sized>(func: Box<P>) -> impl Future<Output = ()> {
    func.call()
}

fn get_boxed_fn() -> Box<DynAsyncFnBoxed> {
    todo!()
}

async fn cursed_fut() {
    wrap_call(get_boxed_fn()).await;
}

fn observe_fut_not_send() {
    fn assert_send<T: Send>(t: T) -> T {
        t
    }
    assert_send(cursed_fut());
}

fn main() {}
