//@ check-pass
//@ edition: 2021

trait TcpStack {
    type Connection<'a>: Sized where Self: 'a;
    fn connect<'a>(&'a self) -> Self::Connection<'a>;

    async fn async_connect<'a>(&'a self) -> Self::Connection<'a>;
}

fn main() {}
