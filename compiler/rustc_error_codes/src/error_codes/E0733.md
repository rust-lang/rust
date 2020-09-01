An [`async`] function used recursion without boxing.

Erroneous code example:

```edition2018,compile_fail,E0733
async fn foo(n: usize) {
    if n > 0 {
        foo(n - 1).await;
    }
}
```

To perform async recursion, the `async fn` needs to be desugared such that the
`Future` is explicit in the return type:

```edition2018,compile_fail,E0720
use std::future::Future;
fn foo_desugared(n: usize) -> impl Future<Output = ()> {
    async move {
        if n > 0 {
            foo_desugared(n - 1).await;
        }
    }
}
```

Finally, the future is wrapped in a pinned box:

```edition2018
use std::future::Future;
use std::pin::Pin;
fn foo_recursive(n: usize) -> Pin<Box<dyn Future<Output = ()>>> {
    Box::pin(async move {
        if n > 0 {
            foo_recursive(n - 1).await;
        }
    })
}
```

The `Box<...>` ensures that the result is of known size, and the pin is
required to keep it in the same place in memory.

[`async`]: https://doc.rust-lang.org/std/keyword.async.html
