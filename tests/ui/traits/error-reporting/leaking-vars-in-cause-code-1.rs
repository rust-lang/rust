//@ compile-flags: -Znext-solver
//@ edition: 2024
//
// A regression test for the ICE variant in trait-system-refactor-initiative#245.
// We'll meet regions that're already popped off when using parent predicate in cause code.
// `cause` in `Obligation` is ignored by folders/visitors.
// In this case, `fudge_inference_if_ok` doesn't fudge a region var in cause code.
//
// The old solver doesn't trigger ICE because regions in the predicate are replaced with
// placeholders when checking generator witness. Besides, the old solver doesn't eagerly
// resolves vars before canonicalizing the predicate in `predicate_must_hold_modulo_regions`.

trait AsyncFn: Send + 'static {
    type Fut: Future<Output = ()> + Send;

    fn call(&self) -> Self::Fut;
}

async fn wrap_call<P: AsyncFn + ?Sized>(filter: &P) {
    filter.call().await;
}

fn get_boxed_fn() -> Box<DynAsyncFnBoxed> {
    todo!()
}

async fn cursed_fut() {
    wrap_call(get_boxed_fn().as_ref()).await;
}

fn observe_fut_not_send() {
    assert_send(cursed_fut());
    //~^ ERROR: `dyn AsyncFn<Fut = Pin<Box<dyn Future<Output = ()> + Send>>>` cannot be shared between threads safely [E0277]
}

fn assert_send<T: Send>(t: T) -> T {
    t
}

pub type BoxFuture<'a, T> = std::pin::Pin<Box<dyn Future<Output = T> + Send + 'a>>;
type DynAsyncFnBoxed = dyn AsyncFn<Fut = BoxFuture<'static, ()>>;

fn main() {}
