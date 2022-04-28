// Regression test for https://github.com/rust-lang/rust/issues/84634
#![crate_name = "foo"]

use std::pin::Pin;
use std::task::Poll;

pub trait Stream {
    type Item;

    fn poll_next(mut self: Pin<&mut Self>) -> Poll<Option<Self::Item>>;
    fn size_hint(&self) -> (usize, Option<usize>);
}

// @has 'foo/trait.Stream.html'
// @has - '//*[@class="code-header in-band"]' 'impl<S> Stream for &mut S'
// @has - '//*[@class="where fmt-newline"]' 'where S: ?Sized + Stream + Unpin'
impl<S: ?Sized + Stream + Unpin> Stream for &mut S {
    type Item = S::Item;

    fn poll_next(
        mut self: Pin<&mut Self>,
    ) -> Poll<Option<Self::Item>> {
        S::poll_next(Pin::new(&mut **self), cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }
}
