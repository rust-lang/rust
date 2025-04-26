//@ check-pass
//@ edition:2018

#![allow(non_fmt_panics, unreachable_code)]

use std::fmt::{self, Display};
use std::marker::PhantomData;

struct NotSend {
    marker: PhantomData<*const u8>,
}

const NOT_SEND: NotSend = NotSend { marker: PhantomData };

impl Display for NotSend {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("this value does not implement Send")
    }
}

async fn f(_: u8) {}

// Exercises this matcher in panic_2015:
// ($fmt:expr, $($arg:tt)+) => $crate::panicking::panic_fmt(...)
async fn panic_fmt() {
    // Panic returns `!`, so the await is never reached, and in particular the
    // temporaries inside the formatting machinery are not still alive at the
    // await point.
    let todo = "...";
    f(panic!("not yet implemented: {}", todo)).await;
}

// Exercises ("{}", $arg:expr) => $crate::panicking::panic_display(&$arg)
async fn panic_display() {
    f(panic!("{}", NOT_SEND)).await;
}

// Exercises ($msg:expr) => $crate::panicking::panic_str($msg)
async fn panic_str() {
    f(panic!((NOT_SEND, "...").1)).await;
}

// Exercises ($msg:expr) => $crate::panicking::unreachable_display(&$msg)
async fn unreachable_display() {
    f(unreachable!(NOT_SEND)).await;
}

fn require_send(_: impl Send) {}

fn main() {
    require_send(panic_fmt());
    require_send(panic_display());
    require_send(panic_str());
    require_send(unreachable_display());
}
