// check-pass
// edition: 2021

#![feature(async_fn_in_trait)]
#![allow(incomplete_features)]

trait TcpStack {
    type Connection<'a>: Sized where Self: 'a;
    fn connect<'a>(&'a self) -> Self::Connection<'a>;
    async fn async_connect<'a>(&'a self) -> Self::Connection<'a>;
}

fn main() {}
