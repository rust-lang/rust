//@ edition:2021
// Regression test for #87211.
// Test that we suggest removing `&` from references to futures,
// including let-bindings and parameter types, but not `&dyn Future`.

async fn my_async_fn() {}

async fn foo() {
    let fut = &my_async_fn();
    fut.await; //~ ERROR `&impl Future<Output = ()>` is not a future
}

async fn direct_ref_await() {
    (&my_async_fn()).await; //~ ERROR `&impl Future<Output = ()>` is not a future
}

async fn bar(fut: &impl std::future::Future<Output = ()>) {
    fut.await; //~ ERROR is not a future
}

async fn dyn_ref_param(fut: &dyn std::future::Future<Output = ()>) {
    fut.await; //~ ERROR is not a future
}

async fn typed_let_binding() {
    let fut: &_ = &my_async_fn();
    fut.await; //~ ERROR `&impl Future<Output = ()>` is not a future
}

async fn ref_param_borrowed_expr(fut: &impl std::future::Future<Output = ()>) {
    (&fut).await; //~ ERROR is not a future
}

async fn double_ref_direct() {
    (&&my_async_fn()).await; //~ ERROR is not a future
}

async fn double_ref_let() {
    let fut = &&my_async_fn();
    fut.await; //~ ERROR is not a future
}

fn main() {}
