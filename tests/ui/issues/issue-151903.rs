//@ edition:2021
//! UI test for spurious double borrow error with temporaries in match expressions.
//!
//! This test demonstrates that temporaries created in a match scrutinee
//! extend to the end of the match, causing an E0502 borrow error.
//!
//! See: https://github.com/rust-lang/rust/issues/151903

use std::task::{Context, Poll};
use std::pin::Pin;
use std::future::Future;

struct Foo<'buf> {
    data: &'buf String,
}

async fn read_foo<'buf>(buffer: &'buf mut String) -> Result<Foo<'buf>, ()> {
    Ok(Foo { data: buffer })
}

fn poll_next(mut buffer: Pin<&mut String>, cx: &mut Context<'_>) -> Poll<()> {
    let _ = match Box::pin(read_foo(&mut *buffer)).as_mut().poll(cx) {
        Poll::Ready(Ok(foo)) => {
            println!("buffer: {}", buffer);
//~^ ERROR[E0502]
        }
        Poll::Ready(Err(())) => {}
        Poll::Pending => {}
    };

    Poll::Pending
}

fn main() {}
