// edition:2018
// [nll] check-pass
// revisions: migrate nll
// Explicitly testing nll with revision, so ignore compare-mode=nll
// ignore-compare-mode-nll

#![cfg_attr(nll, feature(nll))]
#![feature(generic_associated_types)]

use std::future::Future;

trait Client {
    type Connecting<'a>: Future + Send
    where
        Self: 'a;

    fn connect(&'_ self) -> Self::Connecting<'_>;
}

fn call_connect<C>(c: &'_ C) -> impl '_ + Future + Send
//[migrate]~^ ERROR the parameter
//[migrate]~| ERROR the parameter
where
    C: Client + Send + Sync,
{
    async move { c.connect().await }
}

fn main() {}
