//@ revisions: bpass1 bpass2
//@ edition: 2024
//@ compile-flags: --emit=obj
//@ ignore-backends: gcc

use std::future::{Future, ready};
use std::pin::Pin;
use std::task::{Context, Poll};

struct Join<I>(I);

impl<I: IntoIterator> Future for Join<I> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
        loop {}
    }
}

fn spawn(_: impl Future + Send) {}

async fn bar() {
    Join([ready(())].into_iter().map(async |ready| ready.await)).await;
}

fn main() {
    spawn(async {
        #[cfg(bpass1)]
        let _ = bar().await;

        #[cfg(bpass2)]
        let result = bar().await;
    });
}
