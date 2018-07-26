// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![allow(dead_code, unused)]
#![feature(futures_api)]

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
