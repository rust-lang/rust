//@ check-pass
//@ edition:2021

use std::fmt::{self, Display};
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

struct AsyncStdout;

impl AsyncStdout {
    fn write_fmt<'a>(&'a mut self, _args: fmt::Arguments) -> WriteFmtFuture<'a, Self>
    where
        Self: Unpin,
    {
        WriteFmtFuture(self)
    }
}

struct WriteFmtFuture<'a, T>(&'a mut T);

impl<'a, T> Future for WriteFmtFuture<'a, T> {
    type Output = io::Result<()>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        unimplemented!()
    }
}

async fn async_main() {
    let _write = write!(&mut AsyncStdout, "...").await;
    let _writeln = writeln!(&mut AsyncStdout, "...").await;
}

fn main() {
    let _ = async_main;
}
