//@ check-pass

#![feature(fn_delegation)]
#![allow(incomplete_features)]
#![feature(staged_api)]
#![unstable(feature = "foo", issue = "none")]

pub mod m {
    #[unstable(feature = "foo", issue = "none")]
    pub struct W<I, F> {
        pub inner: std::iter::Map<I, F>,
    }

    #[unstable(feature = "foo", issue = "none")]
    impl<B, I: Iterator, F: FnMut(I::Item) -> B> Iterator for W<I, F> {
        type Item = B;
        reuse Iterator::{next, fold} { self.inner }
    }
}

fn main() {}
