//@ build-pass (FIXME(62277): could be check-pass?)

#![allow(dead_code, unused)]

use std::task::Poll;

struct K;
struct E;

fn as_result() -> Result<(), E> {
    // From Result
    let K = Ok::<K, E>(K)?;

    // From Poll<Result>
    let _: Poll<K> = Poll::Ready::<Result<K, E>>(Ok(K))?;

    // From Poll<Option<Result>>
    let _: Poll<Option<K>> = Poll::Ready::<Option<Result<K, E>>>(None)?;

    Ok(())
}

fn as_poll_result() -> Poll<Result<(), E>> {
    // From Result
    let K = Ok::<K, E>(K)?;

    // From Poll<Result>
    let _: Poll<K> = Poll::Ready::<Result<K, E>>(Ok(K))?;

    // From Poll<Option<Result>>
    let _: Poll<Option<K>> = Poll::Ready::<Option<Result<K, E>>>(None)?;

    Poll::Ready(Ok(()))
}

fn as_poll_option_result() -> Poll<Option<Result<(), E>>> {
    // From Result
    let K = Ok::<K, E>(K)?;

    // From Poll<Result>
    let _: Poll<K> = Poll::Ready::<Result<K, E>>(Ok(K))?;

    // From Poll<Option<Result>>
    let _: Poll<Option<K>> = Poll::Ready::<Option<Result<K, E>>>(None)?;

    Poll::Ready(Some(Ok(())))
}

fn main() {
}
