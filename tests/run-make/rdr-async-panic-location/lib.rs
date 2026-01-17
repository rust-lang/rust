pub async fn async_panic() {
    inner_panic().await;
}

async fn inner_panic() {
    panic!("panic inside async fn at lib.rs:6");
}

pub async fn nested_async_panic() {
    let fut = async {
        panic!("panic in async block at lib.rs:11");
    };
    fut.await;
}
